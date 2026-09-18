"""Scene-TTO parameter selection; no change to feature values or module modes."""
import math


def validate_fnet_updates(policy):
    if policy not in ('legacy', 'frozen'):
        raise ValueError('Unknown fnet update policy: '+str(policy))


def configure_fnet_updates(model, policy='legacy'):
    """Apply once after initialization and before optimizer construction.

    Legacy preserves existing flags. Frozen clears fnet gradients and disables
    parameter gradients, but leaves coordinate sampling gradients and train/eval
    modes alone. This is not a feature cache or an optimizer-resume operation.
    """
    validate_fnet_updates(policy)
    model.fnet_update_policy = policy
    if policy == 'frozen':
        for parameter in model.fnet.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    return dict(policy=policy,
        trainable_parameter_names=[name for name,p in model.named_parameters() if p.requires_grad],
        trainable_numel=sum(p.numel() for p in model.parameters() if p.requires_grad),
        fnet_trainable_numel=sum(p.numel() for p in model.fnet.parameters() if p.requires_grad))


def validate_fnet_lr(scale, policy='legacy'):
    validate_fnet_updates(policy)
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or not 0 < float(scale) <= 1:
        raise ValueError('fnet_lr_scale must be finite and in (0, 1]')
    if policy == 'frozen' and float(scale) != 1:
        raise ValueError('Frozen fnet cannot also request a scaled learning rate')


def configure_fnet_optimizer(model, scale=1.0):
    """Return parameters and serializable groups; scale=1 retains one old group.

    Other trainable modules retain the base learning rate. Groups only change
    optimizer scheduling; requires_grad, module modes and gradient clipping do not.
    """
    validate_fnet_lr(scale, getattr(model, 'fnet_update_policy', 'legacy'))
    scale = float(scale)
    named = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if scale == 1:
        params = [p for _,p in named]
        groups = [dict(name='all', lr_scale=1.0, parameter_names=[n for n,_ in named])]
    else:
        partitions = [('other', 1.0, [(n,p) for n,p in named if not n.startswith('fnet.')]),
                      ('fnet', scale, [(n,p) for n,p in named if n.startswith('fnet.')])]
        if any(not pairs for _,_,pairs in partitions):
            raise ValueError('Scaled fnet optimizer needs both fnet and other trainable parameters')
        params = [dict(params=[p for _,p in pairs]) for _,_,pairs in partitions]
        groups = [dict(name=name, lr_scale=lr_scale, parameter_names=[n for n,_ in pairs])
                  for name,lr_scale,pairs in partitions]
    model.fnet_lr_scale = scale
    model.optimizer_parameter_groups = groups
    return params, groups


def validate_fnet_checkpoint(checkpoint, policy='legacy', lr_scale=1.0, warmup_steps=0):
    validate_fnet_updates(policy)
    validate_fnet_lr(lr_scale, policy)
    if checkpoint.get('fnet_update_policy','legacy') != policy:
        raise ValueError('Checkpoint fnet update policy differs from run configuration')
    if checkpoint.get('fnet_lr_scale', 1.0) != float(lr_scale):
        raise ValueError('Checkpoint fnet learning rate scale differs from run configuration')
    validate_fnet_warmup(warmup_steps, policy)
    if checkpoint.get('fnet_warmup_steps', 0) != warmup_steps:
        raise ValueError('Checkpoint fnet warmup differs from run configuration')
    if warmup_steps:
        expected = 'head_only' if checkpoint['optimizer_steps'] <= warmup_steps else 'joint'
        if checkpoint.get('fnet_stage') != expected:
            raise ValueError('Checkpoint fnet stage differs from completed update')


def validate_fnet_warmup(steps, policy='legacy'):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError('fnet_warmup_steps must be a nonnegative integer')
    if steps and policy != 'legacy':
        raise ValueError('Staged fnet warmup cannot combine with permanent freezing')


def configure_fnet_warmup(model, steps=0):
    """Call AFTER building the optimizer and its clipping parameter list.

    All eventually trainable parameters stay in their original optimizer groups.
    Frozen parameters have grad=None: AdamW skips both updates and weight decay,
    and starts their moment/step state only when they actually receive gradients.
    """
    validate_fnet_warmup(steps, getattr(model, 'fnet_update_policy', 'legacy'))
    model.fnet_warmup_steps = steps
    if steps:
        model._fnet_staged_parameters = tuple(p for p in model.fnet.parameters() if p.requires_grad)
        if not model._fnet_staged_parameters:
            raise ValueError('Staged warmup requires an initially trainable fnet')
        apply_fnet_stage(model, 0)


def apply_fnet_stage(model, update_step):
    """Set flags for the indicated update (or saved completed step); 0 is initial."""
    steps = getattr(model, 'fnet_warmup_steps', 0)
    if not steps:
        return
    frozen = update_step <= steps
    model.fnet_stage = 'head_only' if frozen else 'joint'
    for parameter in model._fnet_staged_parameters:
        parameter.requires_grad_(not frozen)
        if frozen:
            parameter.grad = None


def fnet_stage_metadata(model):
    """Leave historical exports unchanged when the new option is disabled."""
    steps = getattr(model, 'fnet_warmup_steps', 0)
    return dict(fnet_warmup_steps=steps, fnet_stage=model.fnet_stage) if steps else {}
