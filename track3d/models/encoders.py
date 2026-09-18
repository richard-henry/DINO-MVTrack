"""Frozen semantic features with an exact legacy DINOv2 path and validated caches.

DINOv3 extraction can run in a separate environment. No dependency upgrade is
required in the existing scene-TTO environment to consume frozen patch features.
"""
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F

FAMILIES={'dinov2':dict(patch_size=14,prefix_tokens=1,model_id='facebook/dinov2-small'),
          'dinov3':dict(patch_size=16,prefix_tokens=5,model_id='facebook/dinov3-vits16-pretrain-lvd1689m')}
CACHE_SCHEMA='track3d_semantic_patches_v1'


def tensor_fingerprint(tensor):
    value=tensor.detach().cpu().contiguous()
    h=hashlib.sha256(f'{str(value.dtype)}:{tuple(value.shape)}:'.encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes());return h.hexdigest()


def feature_geometry(image_size,patch_size):
    h,w=map(int,image_size);eh,ew=(h//patch_size)*patch_size,(w//patch_size)*patch_size
    if min(eh,ew)<patch_size:raise ValueError('Image is smaller than one semantic patch')
    fh,fw=eh//patch_size,ew//patch_size
    # Preserve current projection convention; report physical patch centers separately.
    return dict(image_size=[h,w],encoder_input_size=[eh,ew],feature_size=[fh,fw],patch_size=patch_size,
        image_to_feature_scale=[fw/w,fh/h],mapping='legacy_scaled_intrinsics_v1',
        patch_center_image_scale=[w/fw,h/fh],patch_center_image_offset=[w/fw/2-.5,h/fh/2-.5],
        valid_image_region=[0,0,w,h],preprocessing='imagenet_normalize_then_floor_resize_bilinear_align_corners_false')


def extract_patch_tokens(model,rgbs,patch_size=14,prefix_tokens=1,chunk_size=16):
    """Raw patches [B,S,V,D,Hf,Wf], preserving old op order/dtypes/chunking."""
    if rgbs.ndim!=6 or rgbs.shape[3]!=3 or chunk_size<1:raise ValueError('Expected B,S,V,3,H,W RGB')
    b,s,v,c,h,w=rgbs.shape;geometry=feature_geometry((h,w),patch_size)
    pixels=rgbs.clone().reshape(b*s*v,c,h,w)/255.
    mean=torch.tensor([.485,.456,.406],device=rgbs.device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=rgbs.device).view(1,3,1,1)
    pixels=(pixels-mean)/std;eh,ew=geometry['encoder_input_size'];fh,fw=geometry['feature_size']
    if (h,w)!=(eh,ew):pixels=F.interpolate(pixels,size=(eh,ew),mode='bilinear',align_corners=False)
    maps=[]
    with torch.no_grad():
        for start in range(0,len(pixels),chunk_size):
            output=model(pixel_values=pixels[start:start+chunk_size])
            tokens=output.last_hidden_state[:,prefix_tokens:,:]
            if tokens.shape[1]!=fh*fw:raise ValueError('Patch token count mismatch: check CLS/register configuration')
            maps.append(tokens.reshape(len(tokens),fh,fw,tokens.shape[-1]).permute(0,3,1,2))
    patches=torch.cat(maps,dim=0).reshape(b,s,v,maps[0].shape[1],fh,fw)
    return patches,geometry


def write_feature_cache(path,patches,rgbs,family,encoder_identity,provenance=None):
    if family not in FAMILIES:raise ValueError('Unsupported semantic encoder')
    if not encoder_identity.get('weights_sha256'):raise ValueError('Cache requires encoder weight identity')
    geometry=feature_geometry(rgbs.shape[-2:],FAMILIES[family]['patch_size'])
    expected=(*rgbs.shape[:3],*geometry['feature_size'])
    if patches.ndim!=6 or (*patches.shape[:3],*patches.shape[-2:])!=expected:raise ValueError('Feature and RGB geometry differ')
    if not patches.is_floating_point() or not torch.isfinite(patches).all():raise ValueError('Invalid patch features')
    metadata=dict(schema=CACHE_SCHEMA,family=family,encoder=encoder_identity,geometry=geometry,
        rgb_sha256=tensor_fingerprint(rgbs.float()),rgb_shape=list(rgbs.shape),
        features_sha256=tensor_fingerprint(patches),features_shape=list(patches.shape),
        features_dtype=str(patches.dtype),normalized=False,provenance=provenance or {})
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:torch.save(dict(metadata=metadata,patches=patches.detach().cpu()),stream)
    return metadata


class FrozenFeatureCache:
    """CPU-resident non-parameter cache, validated against RGB content every use."""
    def __init__(self,path,family):
        if family not in FAMILIES:raise ValueError('Unsupported semantic encoder')
        self.path=str(Path(path).resolve())
        # Only tensors and primitive metadata; no arbitrary Python objects needed.
        payload=torch.load(self.path,map_location='cpu',weights_only=True)
        self.metadata=payload['metadata'];self.patches=payload['patches']
        m=self.metadata
        if m['schema']!=CACHE_SCHEMA or m['family']!=family or m['normalized']:
            raise ValueError('Semantic cache family/schema mismatch')
        if not m['encoder'].get('weights_sha256'):raise ValueError('Missing encoder fingerprint')
        if m['features_sha256']!=tensor_fingerprint(self.patches):raise ValueError('Semantic feature content mismatch')
        if list(self.patches.shape)!=m['features_shape'] or str(self.patches.dtype)!=m['features_dtype']:
            raise ValueError('Semantic feature shape/dtype mismatch')
        shape=m['rgb_shape'];geometry=feature_geometry(shape[-2:],FAMILIES[family]['patch_size'])
        if m['geometry']!=geometry:raise ValueError('Semantic spatial transform mismatch')
        if self.patches.ndim!=6 or list(self.patches.shape[:3])!=shape[:3] or list(self.patches.shape[-2:])!=geometry['feature_size']:
            raise ValueError('Semantic cache dimensions mismatch')
        if not self.patches.is_floating_point() or not torch.isfinite(self.patches).all():raise ValueError('Invalid cached features')
        self.sha256=hashlib.sha256(Path(self.path).read_bytes()).hexdigest()

    def get(self,rgbs):
        if list(rgbs.shape)!=self.metadata['rgb_shape'] or tensor_fingerprint(rgbs.float())!=self.metadata['rgb_sha256']:
            raise ValueError('Semantic cache RGB/frame/view order mismatch')
        return self.patches.to(device=rgbs.device),self.metadata['geometry']

    def identity(self):
        return dict(path=self.path,sha256=self.sha256,metadata=self.metadata)


def semantic_identity(model):
    result = dict(family=getattr(model,'semantic_encoder','dinov2'),
        cache=model.semantic_cache.identity() if getattr(model,'semantic_cache',None) is not None else None,
        mapping=('dino_physical_patch_center_v1' if getattr(model,'correlation_geometry','legacy') == 'dino_patch_center'
                 else 'legacy_scaled_intrinsics_v1'),
        correlation_geometry=getattr(model,'correlation_geometry','legacy'),
        dino_query_policy=getattr(model,'dino_query_policy','legacy'),
        view_weight_policy=getattr(model,'view_weight_policy','legacy'))
    # Preserve legacy serialized identity exactly; opt-in policies must be explicit.
    if getattr(model,'coordinate_update_policy','legacy')!='legacy':
        result['coordinate_update_policy']=model.coordinate_update_policy
    if getattr(model,'score_fusion_policy','legacy')!='legacy':
        result['score_fusion_policy']=model.score_fusion_policy
    return result


def load_common_initialization(model,path):
    """Copy all common fnet/fusion/projection parameters, excluding semantic backbone."""
    payload=torch.load(path,map_location='cpu',weights_only=False)
    source=payload['model_state_dict'];current=model.state_dict()
    names=[name for name in current if name.split('.')[0] in ('fnet','fusion_net','dino_proj')]
    if not names:raise ValueError('No shared model parameters')
    for name in names:
        if name not in source or source[name].shape!=current[name].shape:raise ValueError('Incompatible common initialization: '+name)
    model.load_state_dict({**current,**{name:source[name] for name in names}},strict=True)
    return dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),keys=names)


def validate_checkpoint_encoder(checkpoint, model):
    """Reject a changed cache/family on replay; permit moving identical cache files."""
    actual = semantic_identity(model)
    expected = checkpoint.get('semantic_identity')
    if (expected or {}).get('score_fusion_policy','legacy') != getattr(model,'score_fusion_policy','legacy'):
        raise ValueError('Checkpoint score fusion policy differs')
    if (expected or {}).get('coordinate_update_policy','legacy') != getattr(model,'coordinate_update_policy','legacy'):
        raise ValueError('Checkpoint coordinate update policy differs')
    if (expected or {}).get('view_weight_policy','legacy') != actual['view_weight_policy']:
        raise ValueError('Checkpoint view weight policy differs')
    if (expected or {}).get('dino_query_policy','legacy') != actual['dino_query_policy']:
        raise ValueError('Checkpoint DINO query policy differs')
    if expected is None:
        if actual['family'] != 'dinov2' or actual['cache'] is not None or actual['correlation_geometry'] != 'legacy':
            raise ValueError('Legacy checkpoint requires legacy live DINOv2; use explicit common initialization for encoder A/B')
        return
    if expected.get('correlation_geometry','legacy') != actual['correlation_geometry']:
        raise ValueError('Checkpoint correlation geometry differs')
    if expected['family'] != actual['family'] or expected['mapping'] != actual['mapping']:
        raise ValueError('Checkpoint semantic encoder differs')
    old_cache, new_cache = expected['cache'], actual['cache']
    if (old_cache is None) != (new_cache is None):
        raise ValueError('Checkpoint semantic cache mode differs')
    if old_cache is not None and (old_cache['sha256'] != new_cache['sha256'] or old_cache['metadata'] != new_cache['metadata']):
        raise ValueError('Checkpoint semantic cache differs')
