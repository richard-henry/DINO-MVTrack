"""Explicit query-consistent sampling of endpoint-resized correlation maps."""

POLICIES = ('legacy', 'query_aligned', 'dino_patch_center')


def validate_correlation_geometry(policy):
    if policy not in POLICIES:
        raise ValueError('Unknown correlation geometry policy: '+str(policy))


def query_aligned_grid(grid, feature_size, correlation_size, image_size, target_scale):
    """Map shared projected coordinates to an upsampled correlation grid.

    Preserve the existing query lookup u*Wf/W. Because the correlation image
    uses align_corners=True, its lookup must be (u*Wf/W)*(Wc-1)/(Wf-1).
    This fixes path consistency only, not the physical patch-center convention.
    No clamping: out-of-bounds coordinates retain existing padding semantics.
    """
    fh,fw=feature_size
    ch,cw=correlation_size
    ih,iw=image_size
    if min(fh,fw,ch,cw,ih,iw)<=1 or target_scale<=0:
        raise ValueError('Nondegenerate sizes and positive target scale required')
    sx=(fw/iw)/target_scale*(cw-1)/(fw-1)
    sy=(fh/ih)/target_scale*(ch-1)/(fh-1)
    if sx == 1.0 and sy == 1.0:
        return grid
    return grid*grid.new_tensor([sx,sy])


def patch_center_intrinsics(scaled_intrinsics, feature_size, image_size):
    """Translate already resized DINO intrinsics to physical patch centers.

    The floor-resized encoder uses (u+.5)*Wf/W-.5; the existing resized
    intrinsics produce u*Wf/W. Keep the input unchanged and offset both rows.
    """
    fh, fw = feature_size; ih, iw = image_size
    if min(fh, fw, ih, iw) <= 1:
        raise ValueError('Nondegenerate image and feature sizes required')
    result = scaled_intrinsics.clone()
    result[..., 0, :] = result[..., 0, :] + (.5 * fw / iw - .5) * scaled_intrinsics[..., 2, :]
    result[..., 1, :] = result[..., 1, :] + (.5 * fh / ih - .5) * scaled_intrinsics[..., 2, :]
    return result


def patch_center_grid(grid, feature_size, correlation_size, image_size, target_scale):
    """Physical patch coordinate followed by inverse endpoint upsampling.

    Shared grid is target_scale*u. Convert to (u+.5)*Wf/W-.5, then to
    the align_corners=True correlation grid. Do not clamp invalid/border sites.
    This changes DINO only; fnet retains its existing coordinate convention.
    """
    fh, fw = feature_size; ch, cw = correlation_size; ih, iw = image_size
    if min(fh, fw, ch, cw, ih, iw) <= 1 or target_scale <= 0:
        raise ValueError('Nondegenerate sizes and positive target scale required')
    feature = (grid / target_scale + .5) * grid.new_tensor([fw / iw, fh / ih]) - .5
    return feature * grid.new_tensor([(cw - 1) / (fw - 1), (ch - 1) / (fh - 1)])
