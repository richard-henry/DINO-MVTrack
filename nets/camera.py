import torch
import torch.nn.functional as F

def update_after_resize(K, width_ratio, height_ratio):
    # height, width = image_shape
    # new_height, new_width = new_image_shape

    # fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # new_fx = fx * width_ratio
    # new_fy = fy * height_ratio
    # new_cx = cx * width_ratio
    # new_cy = cy * height_ratio

    # K[0, 0], K[1, 1], K[0, 2], K[1, 2] = new_fx, new_fy, new_cx, new_cy
    K[:,0,0] = width_ratio * K[:,0,0]
    K[:,1,1] = height_ratio * K[:,1,1]
    K[:,0,2] = width_ratio * K[:,0,2]
    K[:,1,2] = height_ratio * K[:,1,2]
    return K


def batch_camera_projection(world_points, intrinsics, extrinsics, h=None, w=None):
    """
    input:
    world_points(B,S,N,3):world coordinates of 3d points
    intrinsics(B,S,V,3,3):camera intrinsics
    extrinsics(B,S,V,4,4):camera extrinsics

    output:
    pixel_coords(B,S,V,N,2):pixel coordinates of points
    depths(B,S,V,N,1):depths of points
    rays(B,S,V,N,3):Rays of points
    masks(B,S,V,N,1):masks of points
    """
    B, S, N, _ = world_points.shape
    _, _, V, _, _ = intrinsics.shape
    
    # 扩展世界坐标点为齐次坐标 [B, S, N, 4]
    ones = torch.ones(B, S, N, 1, device=world_points.device, dtype=world_points.dtype)
    world_points_homo = torch.cat([world_points, ones], dim=-1)  # [B, S, N, 4]
    
    # 重新调整形状以便批量处理 [B, S, V, N, 4]
    world_points_expanded = world_points_homo.unsqueeze(2).expand(B, S, V, N, 4)
    
    # 重新调整外参矩阵形状 [B*S*V, 4, 4]
    extrinsics_flat = extrinsics.reshape(-1, 4, 4)
    world_points_flat = world_points_expanded.reshape(-1, N, 4).transpose(1, 2)
    
    # 世界坐标到相机坐标的转换 [B*S*V, 4, N]
    camera_coords_homo_flat = torch.bmm(extrinsics_flat, world_points_flat)
    camera_coords_homo_flat = camera_coords_homo_flat.transpose(1, 2)  # [B*S*V, N, 4]
    
    # 提取3D坐标和深度
    camera_coords_flat = camera_coords_homo_flat[..., :3]  # [B*S*V, N, 3]
    rays = F.normalize(camera_coords_flat, p=2, dim=-1)

    # camera_coords_flat = torch.randn(camera_coords_flat.shape).to(camera_coords_flat.device)
    # camera_coords_flat[:,0:N//2, 2:3] = 0.0
    # #print('camera_coords_flat:',camera_coords_flat)


    depths_flat = camera_coords_flat[..., 2:3]  # [B*S*V, N, 1]
    #print('depths_flat:',depths_flat)
    d_flat = depths_flat.reshape(-1,1)
    invalid_mask1 = d_flat[:,0]<=0.0
    deep_flat = torch.where(depths_flat>0.0,depths_flat,1.0)

    #print('deep_flat:',deep_flat)
    
    # 归一化到图像平面
    #eps = 1e-8
    
    normalized_coords_flat = camera_coords_flat / deep_flat
    #print('normalized_coords_flat:',normalized_coords_flat)
    
    # 重新调整内参矩阵形状 [B*S*V, 3, 3]
    intrinsics_flat = intrinsics.reshape(-1, 3, 3)
    
    # 应用内参矩阵 [B*S*V, N, 3]
    normalized_coords_flat_transposed = normalized_coords_flat.transpose(1, 2)
    pixel_coords_homo_flat = torch.bmm(intrinsics_flat, normalized_coords_flat_transposed)
    pixel_coords_homo_flat = pixel_coords_homo_flat.transpose(1, 2)  # [B*S*V, N, 3]
    
    # 提取2D像素坐标
    pixel_coords_flat = pixel_coords_homo_flat[..., :2]  # [B*S*V, N, 2]
    
    #提取mask

    #print('pixel_coords_flat:',pixel_coords_flat)

    pixel_flat = pixel_coords_flat.reshape(-1,2)
    # Broadcasted boolean index_put can hit a CUDA internal assertion under
    # strict deterministic algorithms (observed during V02A). Keep the same
    # invalid-depth value and zero gradient with an out-of-place mask.
    pixel_flat = torch.where(invalid_mask1[:, None], torch.full_like(pixel_flat, -1.0), pixel_flat)
    pixel_coords_flat = pixel_flat.reshape(B*S*V,N,2)

    #print('pixel_flat:',pixel_flat)
    #print('pixel_coords_flat:',pixel_coords_flat)
    
    invalid_mask2 = pixel_flat[:, 0] < 0.0
    invalid_mask4 = pixel_flat[:, 1] < 0.0
    if w is not None and h is not None:
        invalid_mask3 = pixel_flat[:, 0] >= w
        invalid_mask5 = pixel_flat[:, 1] >= h
        invalid_mask = invalid_mask1 | invalid_mask2 | invalid_mask3 | invalid_mask4 | invalid_mask5
    else:
        invalid_mask = invalid_mask1 | invalid_mask2 | invalid_mask4
    
    valid_mask = ~invalid_mask


    # 恢复原始形状
    pixel_coords = pixel_coords_flat.reshape(B, S, V, N, 2)
    #print('pixel_coords:',pixel_coords)
    depths = depths_flat.reshape(B, S, V, N, 1)
    rays = rays.reshape(B,S,V,N,3)
    mask = valid_mask.reshape(B,S,V,N,1).float()
    
    return pixel_coords, depths, rays, mask

