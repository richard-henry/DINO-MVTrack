"""Export frozen semantic patches in an isolated Transformers environment.

Input is a tensor-only RGB bundle in B,S,V,3,H,W order, produced from the exact
TTO sample by prepare_semantic_rgb. No dataset or tracking dependencies imported.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from track3d.models.encoders import FAMILIES, extract_patch_tokens, write_feature_cache
from track3d.utils.reproducibility import configure_numerics, numerical_settings, state_digest


def export(bundle, output, family='dinov3', model_path=None, revision=None,
           device='cuda', precision='bf16', chunk_size=16, native_repo=None):
    if Path(output).exists():
        raise FileExistsError(output)
    configure_numerics('strict')
    spec = FAMILIES[family]
    source = model_path or spec['model_id']
    if native_repo:
        if family != 'dinov3' or not model_path or revision:
            raise ValueError('Native extraction needs family=dinov3 and a local checkpoint; source revision is locked')
        from track3d.models.dinov3_native import load_native_dinov3
        model, encoder = load_native_dinov3(native_repo, model_path)
        prefix = encoder['prefix_tokens']
        model = model.to(device).eval()
    else:
        import transformers
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(source, revision=revision)
        expected_type = {'dinov2': 'dinov2', 'dinov3': 'dinov3_vit'}[family]
        if config.model_type != expected_type or config.patch_size != spec['patch_size']:
            raise ValueError('Model configuration does not match selected encoder family')
        prefix = 1 + getattr(config, 'num_register_tokens', 0)
        if prefix != spec['prefix_tokens'] or config.hidden_size != 384:
            raise ValueError('A1 supports the registered small encoder only')
        model = AutoModel.from_pretrained(source, revision=revision).to(device).eval()
        encoder = dict(backend='transformers', model_id=source, revision=revision,
                       resolved_commit=getattr(config, '_commit_hash', None),
                       weights_sha256=state_digest(model.state_dict()),
                       config=config.to_dict(), transformers_version=str(transformers.__version__),
                       torch_version=str(torch.__version__), prefix_tokens=prefix)
    model.requires_grad_(False)
    encoder.update(precision=precision, chunk_size=chunk_size)
    payload = torch.load(bundle, map_location='cpu', weights_only=True)
    if payload['schema'] != 'track3d_semantic_rgb_v1':
        raise ValueError('Unsupported RGB bundle')
    rgbs = payload['rgbs'].float().to(device)
    is_cuda = torch.device(device).type == 'cuda'
    if precision != 'float32' and not is_cuda:
        raise ValueError('Mixed precision extraction requires CUDA')
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    with torch.amp.autocast(torch.device(device).type, enabled=precision != 'float32',
                            dtype=torch.bfloat16 if precision == 'bf16' else torch.float16):
        patches, _ = extract_patch_tokens(model, rgbs, spec['patch_size'], prefix, chunk_size)
    if is_cuda:
        torch.cuda.synchronize()
    extraction_seconds = time.monotonic() - start
    provenance = dict(payload.get('provenance', {}), extraction_seconds=extraction_seconds,
                      numerical_settings=numerical_settings(),
                      peak_cuda_bytes=torch.cuda.max_memory_allocated() if is_cuda else 0)
    metadata = write_feature_cache(output, patches, rgbs, family, encoder, provenance)
    print(json.dumps(dict(output=str(Path(output).resolve()),
                          features_shape=metadata['features_shape'],
                          extraction_seconds=extraction_seconds), indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--family', choices=FAMILIES, default='dinov3')
    parser.add_argument('--model-path', help='Authorized local HF snapshot directory or model ID')
    parser.add_argument('--native-repo', help='Pinned official DINOv3 checkout for a native .pth checkpoint')
    parser.add_argument('--revision', help='Pin an official model commit when downloading')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=['float32', 'bf16', 'float16'], default='bf16')
    parser.add_argument('--chunk-size', type=int, default=16)
    args = parser.parse_args()
    export(args.rgb_bundle, args.output, args.family, args.model_path, args.revision,
           args.device, args.precision, args.chunk_size, args.native_repo)


if __name__ == '__main__':
    main()
