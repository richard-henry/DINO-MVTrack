"""Input-defined reprojection support, including off-screen predictions."""
import torch


CONTRACT = dict(version='fixed_support_v1', depth_floor_fraction=0.05,
                depth_scale='detached_positive_initial_depth_median_per_camera',
                depth_penalty='mean_abs_focal_times_relu(floor-z)/depth_scale',
                support='fixed_query_visibility_times_pseudo_visibility_times_causal_mask',
                projection='xy/max(z,floor), principal_point_preserved',
                loss='input_mass_normalized_L1_plus_depth_penalty')


def camera_points(world, extrinsics):
    points = torch.cat((world, torch.ones_like(world[..., :1])), dim=-1)
    return torch.einsum('bsvij,bsnj->bsvni', extrinsics, points)[..., :3]


def input_depth_scale(reference, extrinsics, mask):
    with torch.no_grad():
        z = camera_points(reference.detach(), extrinsics)[..., 2]
        scales = []
        for batch in range(z.shape[0]):
            views = []
            for view in range(z.shape[2]):
                values = z[batch, :, view]
                positive = torch.isfinite(values) & (values > 0)
                supported = positive & (mask[batch, :, view] > 0)
                pool = values[supported] if supported.any() else values[positive]
                if pool.numel() == 0:
                    raise ValueError('fixed_support requires positive input depth in each camera')
                views.append(pool.median())
            scales.append(torch.stack(views))
        return torch.stack(scales)[:, None, :, None]


def fixed_support_terms(world, intrinsics, extrinsics, scale):
    """Return finite projection and a pixel-scaled positive-depth penalty.

    Depth scale comes only from the immutable initialization, never predictions.
    Ordinary positive depths retain the perspective projection. Clamping below
    the input-scaled near plane leaves XY gradients; the barrier restores Z.
    """
    xyz = camera_points(world, extrinsics)
    z = xyz[..., 2]
    floor = CONTRACT['depth_floor_fraction'] * scale
    xy = xyz[..., :2] / torch.maximum(z, floor)[..., None]
    homogeneous = torch.cat((xy, torch.ones_like(z[..., None])), dim=-1)
    uv = torch.einsum('bsvij,bsvnj->bsvni', intrinsics, homogeneous)[..., :2]
    focal = (intrinsics[..., 0, 0].abs() + intrinsics[..., 1, 1].abs()) * .5
    barrier = focal[..., None] * torch.relu(floor-z) / scale
    return uv, barrier
