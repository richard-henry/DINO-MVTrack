"""Explicit GT-assisted reprojection-only diagnostics; never a deployable policy."""
import torch

POLICIES = ('oracle_coordinates', 'oracle_coordinates_visibility')
CONTRACT = dict(version='oracle_reprojection_v1', diagnostic_only=True,
    input_protocol='GT_coordinates_for_loss_only',
    changed_component='reprojection_targets_and_optionally_its_visibility',
    unchanged='initialization_search_forward_visibility_rigid_targets_query_view_mask',
    invalid_supported_gt='error', project='unclipped_float32_positive_depth')


def oracle_targets(world, visibility, intrinsics, extrinsics, pseudo_visibility,
                   query_visibility, query_t, policy):
    if policy not in POLICIES:
        raise ValueError('Unknown oracle diagnostic policy')
    with torch.no_grad(), torch.amp.autocast(world.device.type, enabled=False):
        xyz = world.detach().float()
        k = intrinsics.detach().float(); rt = extrinsics.detach().float()
        if xyz.ndim != 4 or xyz.shape[-1] != 3:
            raise ValueError('Expected GT tracks [B,S,N,3]')
        b, s, n, _ = xyz.shape; v = k.shape[2]
        if (k.shape != (b,s,v,3,3) or rt.shape != (b,s,v,4,4)
                or visibility.shape != (b,s,n,v) or pseudo_visibility.shape != (b,s,v,n)
                or query_visibility.shape != (b,1,n,v) or query_t.shape != (b,n)):
            raise ValueError('Oracle geometry/visibility dimensions differ')
        hom = torch.cat((xyz, torch.ones_like(xyz[..., :1])), -1)
        camera = torch.einsum('bsvij,bsnj->bsvni', rt, hom)[..., :3]
        image = torch.einsum('bsvij,bsvnj->bsvni', k, camera)
        valid = (camera[..., 2] > 1e-6) & (image[..., 2].abs() > 1e-6)
        valid &= torch.isfinite(image).all(-1)
        target = image[..., :2] / torch.where(valid, image[..., 2], 1.)[..., None]
        target = torch.where(valid[..., None], target, torch.zeros_like(target))
        mask = (pseudo_visibility.detach().clone() if policy == 'oracle_coordinates'
                else visibility.detach().permute(0,1,3,2).float())
        if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
            raise ValueError('Invalid oracle visibility weights')
        supported = ((mask > 0) & (query_visibility.permute(0,1,3,2) > 0)
            & (torch.arange(s, device=xyz.device).view(1,s,1,1) >= query_t.view(b,1,1,n)))
        if (supported & ~valid).any():
            raise ValueError('Oracle would supervise an undefined/nonpositive GT projection')
        return target, mask
