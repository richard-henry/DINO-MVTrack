"""V01g: native bilinear forward with a bounded, deterministic first backward.

Only finite, int32-safe 2D grids, zero padding and float32/float64 are supported. CUDA
backward requires strict deterministic algorithms: scatter_add_ then uses
PyTorch's deterministic implementation. No feature or coordinate gradients
are detached. Higher-order derivatives are deliberately unsupported.
"""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


SAMPLING_POLICIES = ('native', 'deterministic')


def validate_sampling_policy(policy):
    if policy not in SAMPLING_POLICIES:
        raise ValueError(f'Unknown sampling_policy: {policy!r}')


def sample_2d(input, grid, mode='bilinear', padding_mode='zeros',
              align_corners=True, sampling_policy='native'):
    validate_sampling_policy(sampling_policy)
    if sampling_policy == 'native':
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                             align_corners=align_corners)
    if mode != 'bilinear' or padding_mode != 'zeros' or align_corners not in (True, False):
        raise ValueError('Deterministic sampling supports bilinear/zeros and explicit align_corners only')
    # Match grid_sample CUDA autocast-to-float32; preserve float64 for gradcheck.
    if input.is_cuda and torch.is_autocast_enabled():
        input = input.float() if input.dtype in (torch.float16, torch.bfloat16) else input
        grid = grid.float() if grid.dtype in (torch.float16, torch.bfloat16) else grid
    if input.ndim != 4 or grid.ndim != 4 or grid.shape[-1] != 2 or input.shape[0] != grid.shape[0]:
        raise ValueError('Expected input B,C,H,W and grid B,Hout,Wout,2')
    if input.dtype not in (torch.float32, torch.float64) or grid.dtype != input.dtype or grid.device != input.device:
        raise ValueError('Deterministic sampling requires matching float32/float64 tensors')
    # Native CUDA can return NaN when finite coordinates overflow its index math.
    safe_normalized_limit = 2**30 / max(1, *input.shape[-2:])
    if min(input.shape) <= 0 or not (torch.isfinite(grid) & (grid.abs() <= safe_normalized_limit)).all():
        raise ValueError('Deterministic sampling requires nonempty inputs and finite, int32-safe grids')
    with torch.autocast(device_type=input.device.type, enabled=False):
        return _Bilinear.apply(input, grid, bool(align_corners))


class _Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, align_corners):
        ctx.save_for_backward(input, grid)
        ctx.align_corners = align_corners
        # Keep the exact old forward (including its boundary conventions).
        return F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros',
                             align_corners=align_corners)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        if input.is_cuda and (not torch.are_deterministic_algorithms_enabled()
                              or torch.is_deterministic_algorithms_warn_only_enabled()):
            raise RuntimeError('Deterministic sampling backward requires strict deterministic algorithms')
        B, C, H, W = input.shape
        Q = grid.shape[1] * grid.shape[2]
        need_input, need_grid = ctx.needs_input_grad[:2]
        grad_input = torch.zeros_like(input, memory_format=torch.contiguous_format) if need_input else None
        grad_grid = torch.empty_like(grid, memory_format=torch.contiguous_format) if need_grid else None
        # Batch chunks cap per-neighbor B*C*Q temporaries; no dense sampling matrix.
        chunk = max(1, 262144 // max(1, C * Q))
        sx = (W - 1) / 2 if ctx.align_corners else W / 2
        sy = (H - 1) / 2 if ctx.align_corners else H / 2
        for start in range(0, B, chunk):
            end = min(B, start + chunk)
            g = grid[start:end].reshape(end-start, Q, 2)
            x = (g[..., 0] + 1) * sx - (0 if ctx.align_corners else 0.5)
            y = (g[..., 1] + 1) * sy - (0 if ctx.align_corners else 0.5)
            # Entirely outside the image has zero value and zero derivative.
            # Clamp before integer conversion to avoid overflow for large grids.
            active = (x >= -1) & (x < W) & (y >= -1) & (y < H)
            x = x.clamp(-2, W + 1)
            y = y.clamp(-2, H + 1)
            x0, y0 = x.floor().long(), y.floor().long()
            dx, dy = x - x0, y - y0
            upstream = grad_output[start:end].reshape(end-start, C, Q)
            offset = torch.arange((end-start)*C, device=input.device).reshape(end-start, C, 1) * (H*W)
            source = input[start:end].reshape(-1) if need_grid else None
            gx = torch.zeros_like(x) if need_grid else None
            gy = torch.zeros_like(y) if need_grid else None
            for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                ix, iy = x0 + ox, y0 + oy
                valid = active & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
                wx, wy = (dx if ox else 1-dx), (dy if oy else 1-dy)
                indices = offset + (iy.clamp(0, H-1)*W + ix.clamp(0, W-1)).unsqueeze(1)
                if need_input:
                    contribution = upstream * (wx * wy * valid).unsqueeze(1)
                    grad_input[start:end].view(-1).scatter_add_(0, indices.reshape(-1), contribution.reshape(-1))
                if need_grid:
                    value = source[indices] * valid.unsqueeze(1)
                    dot = (upstream * value).sum(dim=1)
                    gx.add_(dot * wy * (1 if ox else -1))
                    gy.add_(dot * wx * (1 if oy else -1))
            if need_grid:
                target = grad_grid[start:end].reshape(end-start, Q, 2)
                target[..., 0] = gx * sx
                target[..., 1] = gy * sy
        return grad_input, grad_grid, None
