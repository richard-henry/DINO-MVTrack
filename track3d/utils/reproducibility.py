"""Explicit numerical policy and optional, RNG-neutral optimization traces."""
import hashlib
import json
import os
import random

import numpy as np
import torch


def numerical_settings():
    return {
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'fill_uninitialized_memory': torch.utils.deterministic.fill_uninitialized_memory,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cuda_allocator_config': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        'cpu_threads': torch.get_num_threads(),
        'cpu_interop_threads': torch.get_num_interop_threads(),
    }


def configure_numerics(policy='inherit'):
    """Configure before CUDA initialization; inherit preserves ambient behavior."""
    if policy not in ('inherit', 'strict'):
        raise ValueError(f'Unknown numerical_policy: {policy!r}')
    if policy == 'strict':
        workspace = ':4096:8'
        if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != workspace:
            raise RuntimeError('Configure strict numerics before CUDA initialization; '
                               'CUBLAS_WORKSPACE_CONFIG must be :4096:8')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Keep precision/TF32 and AMP unchanged; record their actual values.
    return dict(policy=policy, **numerical_settings())


def scheduler_horizon(max_iters, scheduler_steps=None):
    horizon = max_iters if scheduler_steps is None else int(scheduler_steps)
    if horizon < max_iters or horizon < 0:
        raise ValueError('scheduler_steps must be nonnegative and >= max_iters')
    return horizon


def state_digest(value):
    """Hash values, tensor dtype/shape and bytes, independent of storage IDs."""
    digest = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous()
            digest.update(f'tensor:{x.dtype}:{tuple(x.shape)}:'.encode())
            digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            digest.update(f'numpy:{x.dtype}:{x.shape}:'.encode())
            digest.update(x.tobytes())
        elif isinstance(x, dict):
            digest.update(b'dict[')
            for key in sorted(x, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(x[key])
            digest.update(b']')
        elif isinstance(x, (tuple, list)):
            digest.update(type(x).__name__.encode() + b'[')
            for item in x:
                visit(item)
            digest.update(b']')
        else:
            digest.update((type(x).__name__ + ':' + repr(x) + ';').encode())

    visit(value)
    return digest.hexdigest()


def gradient_digest(model):
    return state_digest({n: p.grad for n, p in model.named_parameters() if p.requires_grad})


def write_trace(path, step, model, optimizer, scheduler, scaler, **observations):
    """After-update state; observations identify the pre-update forward/gradient."""
    record = dict(
        step=step,
        trainable_parameters=state_digest({n: p for n, p in model.named_parameters() if p.requires_grad}),
        buffers=state_digest(dict(model.named_buffers())),
        optimizer=state_digest(optimizer.state_dict()),
        scheduler=state_digest(scheduler.state_dict()),
        scaler=state_digest(scaler.state_dict()),
        rng=state_digest(dict(torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                              numpy=np.random.get_state(), python=random.getstate())),
        **observations,
    )
    with open(path, 'a') as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')


def compare_traces(first, second):
    """Report first mismatch per field; caller separately checks run identity."""
    with open(first) as stream:
        a = [json.loads(line) for line in stream]
    with open(second) as stream:
        b = [json.loads(line) for line in stream]
    if [r['step'] for r in a] != [r['step'] for r in b] or not a:
        raise ValueError('Trace steps differ or are empty')
    differences = {}
    for x, y in zip(a, b):
        for key in sorted(set(x) | set(y)):
            if x.get(key) != y.get(key) and key not in differences:
                differences[key] = x['step']
    return dict(equal=not differences, steps=len(a), first_difference_by_field=differences)
