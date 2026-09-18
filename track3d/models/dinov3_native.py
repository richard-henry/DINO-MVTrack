"""Load the pinned official DINOv3 definition with a verified local state_dict.

Used only in the Python >=3.10 extraction process; the TTO environment consumes
the existing tensor cache format and does not import the official implementation.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from track3d.utils.reproducibility import state_digest


class NativeDINOv3Adapter(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.eval()

    def forward(self, pixel_values):
        if self.training or self.backbone.training:
            raise ValueError('Frozen native encoder must remain in eval mode')
        with torch.no_grad():
            output = self.backbone.forward_features(pixel_values)
        # Official outputs already passed final LayerNorm, matching HF hidden
        # states. L2 normalization stays in Pips.forward, after cache retrieval.
        tokens = torch.cat([output['x_norm_clstoken'].unsqueeze(1),
                            output['x_storage_tokens'], output['x_norm_patchtokens']], dim=1)
        return SimpleNamespace(last_hidden_state=tokens)


def load_native_dinov3(repo_path, checkpoint_path, lock_path=None):
    if sys.version_info < (3, 10):
        raise RuntimeError('Use the Python >=3.10 extraction environment for native DINOv3')
    lock_path = Path(lock_path) if lock_path else Path(__file__).resolve().parents[2]/'configs/dinov3_native.json'
    lock = json.loads(lock_path.read_text())
    repo, checkpoint = Path(repo_path).resolve(), Path(checkpoint_path).resolve()
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if commit != lock['commit']:
        raise ValueError('DINOv3 source commit differs from registered version')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
        raise ValueError('DINOv3 source checkout must be clean')
    file_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if file_sha != lock['checkpoint_sha256']:
        raise ValueError('DINOv3 checkpoint checksum differs from registered weights')
    existing = sys.modules.get('dinov3')
    if existing is not None and Path(existing.__file__).resolve().parent != repo/'dinov3':
        raise ValueError('A different dinov3 package is already imported')
    sys.path.insert(0, str(repo))
    try:
        from dinov3.hub.backbones import dinov3_vits16
        backbone = dinov3_vits16(pretrained=False)
    finally:
        sys.path.pop(0)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    backbone.load_state_dict(state, strict=True)
    if backbone.embed_dim != lock['hidden_size'] or backbone.patch_size != lock['patch_size']:
        raise ValueError('DINOv3 architecture differs from registered small model')
    if backbone.n_storage_tokens + 1 != lock['prefix_tokens']:
        raise ValueError('DINOv3 register token count differs')
    names = subprocess.check_output(['git','-C',str(repo),'ls-files','-z']).decode().strip('\0').split('\0')
    hashes = {name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in names if (repo/name).is_file()}
    identity = dict(backend='official_native', model_id=lock['constructor'],
                    source_repository=lock['repository'], source_commit=commit,
                    source_files_sha256=hashes, checkpoint_path=str(checkpoint),
                    checkpoint_sha256=file_sha, weights_sha256=state_digest(backbone.state_dict()),
                    strict_load=True, state_entries=len(state), patch_size=lock['patch_size'],
                    prefix_tokens=lock['prefix_tokens'], hidden_size=lock['hidden_size'],
                    torch_version=str(torch.__version__), python_version=sys.version)
    return NativeDINOv3Adapter(backbone), identity
