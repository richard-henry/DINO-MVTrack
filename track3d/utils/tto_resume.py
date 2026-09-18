"""Exact checkpoint continuation restricted to the fixed, single-scene loader."""
import json
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from track3d.utils.reproducibility import numerical_settings, scheduler_horizon
from track3d.utils.fnet_updates import validate_fnet_checkpoint, apply_fnet_stage
from track3d.models.encoders import validate_checkpoint_encoder


def resolve_schedule(max_iters, scheduler_steps=None, scheduler_total_steps=None):
    if scheduler_total_steps is None:
        return scheduler_horizon(max_iters, scheduler_steps)
    total = int(scheduler_total_steps)
    if total != scheduler_total_steps or total < 100 or total < max_iters:
        raise ValueError('Explicit scheduler total must be an integer >= 100 and >= max_iters')
    if scheduler_steps is not None and int(scheduler_steps) != total-100:
        raise ValueError('Explicit total conflicts with legacy scheduler_steps + 100')
    return total-100


def budget_independent_recipe(config):
    recipe=config['trainer'].get('convergence_recipe')
    return recipe is not None and recipe['schedule']['kind'] in ('constant','warm_cosine','loss_plateau')


def carry_selected_checkpoint(source_checkpoint, destination, selected_step):
    """Keep the prior best complete checkpoint and matching prediction on resume."""
    source=Path(source_checkpoint).parent; destination=Path(destination)
    for relative in (f'model-{selected_step:09d}.pth',f'predictions/step-{selected_step:06d}_tracks.npz'):
        old,new=source/relative,destination/relative
        if not old.is_file():raise FileNotFoundError('Missing selected checkpoint artifact: '+str(old))
        if new.exists():raise FileExistsError('Selected checkpoint destination exists: '+str(new))
        new.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,new)


def validate_resume_config(old, new, restart_scheduler=False):
    # Saved JSON represents tuple-valued CLI defaults (e.g. crop_size) as lists.
    new = json.loads(json.dumps(new))
    if old['data'] != new['data']:
        raise ValueError('Resume data configuration differs')
    defaults = dict(correlation_geometry='legacy', dino_query_policy='legacy', view_weight_policy='legacy', coordinate_update_policy='legacy', score_fusion_policy='legacy')
    for key in set(old['model']) | set(new['model']):
        if key in ('run_name', 'semantic_identity'):continue
        if old['model'].get(key,defaults.get(key)) != new['model'].get(key,defaults.get(key)):
            raise ValueError('Resume model configuration differs: '+key)
    defaults = dict(fnet_update_policy='legacy', fnet_lr_scale=1.0, fnet_warmup_steps=0)
    keys = ('lr','grad_acc','seed','optimization_seed','device_ids','scheduler_total_steps',
            'numerical_settings','query_visibility_source','fnet_update_policy','fnet_lr_scale','fnet_warmup_steps')
    for key in keys:
        if key == 'scheduler_total_steps' and (restart_scheduler or (budget_independent_recipe(old) and budget_independent_recipe(new))):continue
        if old['trainer'].get(key,defaults.get(key)) != new['trainer'].get(key,defaults.get(key)):
            raise ValueError('Resume trainer configuration differs: '+key)
    if not restart_scheduler and old['trainer'].get('convergence_recipe') != new['trainer'].get('convergence_recipe'):
        raise ValueError('Resume convergence recipe differs')


def restore_tto_state(path, model, optimizer, scheduler, scaler, config, dataset_length, max_iters, restart_scheduler=False):
    if dataset_length != 1 or config['data']['B'] != 1 or config['trainer']['grad_acc'] != 1:
        raise ValueError('Exact resume supports only one scene, B=1, grad_acc=1')
    old=json.loads((Path(path).parent/'config.yaml').read_text())
    validate_resume_config(old,config,restart_scheduler)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if config['model'].get('supervision_policy') == 'retracking_aux':
        identity = config['model']['retracking_input_sha256']
        if saved.get('retracking_input_sha256') != identity:
            raise ValueError('Resume auxiliary input identity differs')
        model.retracking_input_sha256 = identity
    from track3d.utils.anchor_loss import validate_anchor_checkpoint
    validate_anchor_checkpoint(saved, config['model'].get('anchor_policy','legacy'))
    step=saved['optimizer_steps']
    if not 0 < step < max_iters:raise ValueError('Resume checkpoint must precede requested final step')
    validate_fnet_checkpoint(saved,model.fnet_update_policy,model.fnet_lr_scale,
                             getattr(model, 'fnet_warmup_steps', 0))
    apply_fnet_stage(model, step)
    validate_checkpoint_encoder(saved,model)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    if saved.get('trainable_parameter_names',names)!=names:raise ValueError('Resume trainable parameters differ')
    if saved.get('optimizer_parameter_groups',model.optimizer_parameter_groups)!=model.optimizer_parameter_groups:
        raise ValueError('Resume optimizer groups differ')
    if saved['numerical_settings']!=numerical_settings():raise ValueError('Resume numerical settings differ')
    if not restart_scheduler and not budget_independent_recipe(config) and saved['scheduler_state_dict']['total_steps']!=scheduler.total_steps:
        raise ValueError('Resume scheduler horizon differs')
    model.load_state_dict(saved['model_state_dict'],strict=True)
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    if restart_scheduler:
        scheduler.restart_at(step)
    else:
        scheduler.load_state_dict(saved['scheduler_state_dict'])
    scaler.load_state_dict(saved['scaler_state_dict'])
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(saved['torch_rng_state'])
    torch.cuda.set_rng_state_all(saved['cuda_rng_state_all'])
    np.random.set_state(saved['numpy_rng_state']);random.setstate(saved['python_rng_state'])
    return step
