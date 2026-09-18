import torch
import torch.nn as nn
import torch.nn.functional as F
from track3d.utils.deterministic_sampling import sample_2d
import numpy as np
import cv2

class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.InstanceNorm3d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm3d(out_planes)
            )
    
    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)

    
class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)
    

class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.InstanceNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    

class EncoderDecorder(nn.Module):
    def __init__(self):
        super(EncoderDecorder, self).__init__()

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)

        self.mid_res = Res3DBlock(128, 128)

        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)

        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)

        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        x = self.mid_res(x)

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x) 
        x = x + skip_x2

        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x


class V2VNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(V2VNet, self).__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 3),
            Res3DBlock(16, 32),
        )

        self.encoder_decoder = EncoderDecorder()

        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.output_layer(x)

        return x
    

    

def create_boundingbox(trajs, size, grid_dim=10):
    device = trajs.device
    B,S,N,D = trajs.shape  
    assert(D==3)
    center = trajs.reshape(B*S*N, 1, D).to(device)
    if size.ndim == 3 and size.shape == (B, N, D):
        Size = size.unsqueeze(1).expand(B, S, N, D).reshape(B*S*N, 1, D).to(device)
    elif size.shape == (B, S, N, D):
        Size = size.reshape(B*S*N, 1, D).to(device)
    else:
        raise ValueError('Search size must be [B,N,3] or [B,S,N,3]')
    grid = build_grid(1, grid_dim, device).to(dtype=trajs.dtype)
    grid = (grid - (grid_dim - 1) / 2.0) / ((grid_dim - 1) / 2.0)
    world_volume_batch = center + grid * (Size / 2.0)
    return world_volume_batch.reshape(B, S, N, grid_dim, grid_dim, grid_dim, D)

def World2Grid(world_trajs, size, grid_dim):
    device = world_trajs.device
    B,S,N,D = world_trajs.shape
    assert(D==3)
    world_coords = world_trajs.to(device)
    origin = world_coords[:, 0:1]
    if size.ndim == 3 and size.shape == (B, N, D):
        Size = size.unsqueeze(1).to(device)
    elif size.shape == (B, S, N, D):
        Size = size.to(device)
    else:
        raise ValueError('Search size must be [B,N,3] or [B,S,N,3]')
    grid_coords = (world_coords - origin) * (2.0 / Size)
    grid_coords = grid_coords * ((grid_dim - 1) / 2.0) + (grid_dim - 1) / 2.0
    return grid_coords


def Grid2World(grid_trajs, origin, size, grid_dim):  
    
    device = grid_trajs.device
    B,S,N,D = grid_trajs.shape
    B1,N1,D1 = size.shape
    V = grid_dim
    assert(D==3)
    grid_coords = grid_trajs.clone()
    grid_coords = grid_coords.permute(0,2,1,3).reshape(B*N,S,D)
    world_coords = torch.zeros_like(grid_coords).to(device)
    Size = size.reshape(B*N,-1)
    Origin = origin.reshape(B*N,-1)
    for b in range(B*N):
        scale_matrix = torch.diag(Size[b]/2).to(device)
        grid_coords[b] = (grid_coords[b]-torch.tensor([V-1,V-1,V-1]).to(device)/2)/((V-1)/2)
        world_coords[b] = Origin[b] + torch.matmul(grid_coords[b], scale_matrix)
    world_coords = world_coords.reshape(B,N,S,-1).permute(0,2,1,3)
    return world_coords

def feature_sample_2d(coords, fmaps, mode='bilinear', padding_mode='zeros', align_corners=True, sampling_policy='native'):
    """
    input:
    coords(B,V,N,2):2D pixel coordinates
    fmaps(B,V,C,H,W):feature maps for sampling

    output:
    2d_feature(B,V,N,C):2D sampled feature
    """
    B, V, C, H, W = fmaps.shape
    N = coords.shape[2]
    
    # 默认推荐使用 align_corners=True

    # 根据align_corners设置不同的归一化方式
    if align_corners:
        # align_corners=True 时的归一化
        x_normalized = 2.0 * coords[..., 0] / (W - 1) - 1.0
        y_normalized = 2.0 * coords[..., 1] / (H - 1) - 1.0
    else:
        # align_corners=False 时的归一化
        x_normalized = 2.0 * coords[..., 0] / W - 1.0
        y_normalized = 2.0 * coords[..., 1] / H - 1.0
    
    normalized_coords = torch.stack([x_normalized, y_normalized], dim=-1)
    
    # 重塑并采样
    fmaps_flat = fmaps.reshape(B * V, C, H, W)
    grid_flat = normalized_coords.reshape(B * V, N, 1, 2)
    
    sampled = sample_2d(
        fmaps_flat, grid_flat, sampling_policy=sampling_policy,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )

    sampled = sampled.squeeze(-1).transpose(1, 2).reshape(B, V, N, C)
    return sampled



def unproject(features, k_matrices, rt_matrices, world_volume, latent_dim):
    """
    features:BS*n_view*C*H*W
    k_matrix:BS*n_view*3*3
    rt_matrix:BS*n_view*4*4
    world_volume:B*n_point*S*V*V*V*D(3)
    """
    # project in parallel
    device = features.device
    k_matrices = k_matrices.to(device)
    rt_matrices = rt_matrices.to(device)
    batch_size, n_views, c,h,w = features.shape
    B,N,S,V1,V2,V3,D = world_volume.shape
    grid_dim = V1
    assert(batch_size == B*S)
    assert(D==3)

    world_batch = world_volume.permute(0,2,1,3,4,5,6).reshape(B*S*N*V1*V2*V3,D)
    world_homogeneous = euclidean_to_homogeneous(world_batch)
    world_batch = world_homogeneous.reshape(B*S,1,N*V1*V2*V3,-1,1)
    
    k_matrix= k_matrices.reshape(B*S,n_views,1,3,3)
    rt_matrix= rt_matrices.reshape(B*S,n_views,1,4,4)
    camera_batch = torch.matmul(rt_matrix,world_batch).reshape(B*S*n_views*N*V1*V2*V3,-1)[:,:3]
    z = camera_batch[:,2:]
    invalid_mask1 = z[:,0] <= 0.0   # depth must be larger than 0.0
    z[z[:,0]==0.0,0] = 1.0
    ray_dir = F.normalize(camera_batch,dim=1)
    camera_batch =  (camera_batch / z).reshape(B*S,n_views,N*V1*V2*V3,-1,1)
    pixel_coord_proj = torch.matmul(k_matrix,camera_batch).reshape(B*S*n_views*N*V1*V2*V3,-1)
    # invalid_mask1 = pixel_coord_proj[:, 2] <= 0.0  
    # pixel_coord_proj[pixel_coord_proj[:, 2] == 0.0, 2] = 1.0  # not to divide by zero
    pixel_coord_proj = homogeneous_to_euclidean(pixel_coord_proj)
    invalid_mask2 = pixel_coord_proj[:, 0] < 0.0
    invalid_mask3 = pixel_coord_proj[:, 0] >= w
    invalid_mask4 = pixel_coord_proj[:, 1] < 0.0
    invalid_mask5 = pixel_coord_proj[:, 1] >= h

    invalid_mask = invalid_mask1 | invalid_mask2 | invalid_mask3 | invalid_mask4 | invalid_mask5
    valid_mask = ~invalid_mask

    pixel_coord_proj_transformed = torch.zeros_like(pixel_coord_proj)
    pixel_coord_proj_transformed[:, 0] = 2 * (pixel_coord_proj[:, 0] / w - 0.5)
    pixel_coord_proj_transformed[:, 1] = 2 * (pixel_coord_proj[:, 1] / h - 0.5)
    pixel_coord_proj = pixel_coord_proj_transformed.reshape(B*S*n_views,N,V1*V2*V3,-1)

    # if torch.isnan(pixel_coord_proj).any():
    #     print('camera_batch_nan:',torch.isnan(camera_batch).any())
    #     print('z min:',torch.min(z))
    #     print('z max:',torch.max(z))
    #     print('pixel_coords_nan:',torch.isnan(pixel_coord_proj).any())
    #     quit()

    fmaps = features.reshape(B*S*n_views,c,h,w)
    feature_volume = F.grid_sample(fmaps, pixel_coord_proj, padding_mode ="zeros", mode ='bilinear', align_corners=True)
    # if torch.isnan(feature_volume).any():
    #     print('pixel_coords min:',torch.min(pixel_coord_proj))
    #     print('pixel_coords max:',torch.max(pixel_coord_proj))
    #     print('fmaps min:',torch.min(fmaps))
    #     print('fmaps max:',torch.max(fmaps))
    #     print('fmap_nan:',torch.isnan(fmaps).any())
    #     print('pixel_coords_nan:',torch.isnan(pixel_coord_proj).any())
    #     quit()
    feature_volume = feature_volume.reshape(B,S,n_views,c,N,V1,V2,V3).permute(0,1,2,4,5,6,7,3)
    feature_volume = feature_volume.reshape(B*S*n_views*N*V1*V2*V3,-1)
    feature_volume[invalid_mask] = 0.0

    feature_volume = feature_volume.reshape(B*S,n_views,N,V1,V2,V3,c).permute(0,2,1,6,3,4,5)
    volume_batch = feature_volume.reshape(B*S*N,n_views,c,V1,V2,V3)
    volume_mask = valid_mask.reshape(B*S,n_views,N,V1,V2,V3).permute(0,2,1,3,4,5).reshape(B*S*N,n_views,1,V1,V2,V3)
    volume_depth = z.reshape(B*S,n_views,N,V1,V2,V3).permute(0,2,1,3,4,5).reshape(B*S*N,n_views,1,V1,V2,V3)
    volume_dir = ray_dir.reshape(B*S,n_views,N,V1,V2,V3,3).permute(0,2,1,6,3,4,5).reshape(B*S*N,n_views,3,V1,V2,V3)

    return volume_batch,volume_dir,volume_depth,volume_mask


def expand_batch(x: torch.Tensor, frames_per_batch: int) -> torch.Tensor:
    n = x.shape[0]
    other_dims = x.shape[1:]
    return x.reshape(n // frames_per_batch, frames_per_batch, *other_dims)


def collapse_batch(x: torch.Tensor) -> torch.Tensor:
    n, k = x.shape[:2]
    other_dims = x.shape[2:]
    return x.reshape(n * k, *other_dims)

def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=1, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=1, keepdim=True)
    return mean, var

# adapted from: https://github.com/googleinterns/IBRNet/blob/master/ibrnet/mlp_network.py
class IBRNet_Aggregator(nn.Module):
    def __init__(self, feature_dim: int = 32, anti_alias_pooling: bool = False, kernel_size: int = 1, padding: int = 0, use_temb: bool = False, temb_dim: int = 1280):
        super().__init__()

        self.anti_alias_pooling = anti_alias_pooling
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)

        # turn voxel_depth, rays into encoding to add to features
        self.ray_depth_encoder = nn.Sequential(
            nn.Conv3d(4, feature_dim // 2, kernel_size=kernel_size, padding=padding),
            activation_func,
            nn.Conv3d(feature_dim // 2, feature_dim, kernel_size=kernel_size, padding=padding),
            activation_func,
        )

        # turn time embedding into encoding to add to features
        self.use_temb = use_temb
        if use_temb:
            self.temb_encoder = nn.Sequential(
                nn.Linear(temb_dim, temb_dim // 2),
                activation_func,
                nn.Linear(temb_dim // 2, feature_dim),
                activation_func,
            )

        # shared part of feature/weight encoding
        self.base_fc = nn.Sequential(
            nn.Conv3d(feature_dim * 3, feature_dim * 2, kernel_size=kernel_size, padding=padding),
            activation_func,
            nn.Conv3d(feature_dim * 2, feature_dim, kernel_size=kernel_size, padding=padding),
            activation_func,
        )

        # compute first part of averaging weights, final features
        self.vis_fc = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=kernel_size, padding=padding),
            activation_func,
            nn.Conv3d(feature_dim, feature_dim + 1, kernel_size=kernel_size, padding=padding),
            activation_func,
        )

        # compute second part of averaging weights
        self.vis_fc2 = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, kernel_size=kernel_size, padding=padding),
            activation_func,
            nn.Conv3d(feature_dim, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

        # combine (weight, mean, var) into final grid
        self.statistics_out = nn.Sequential(
            nn.Conv3d(2 * feature_dim + 1, feature_dim, kernel_size=kernel_size, padding=padding),
            activation_func,
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor, voxel_depth: torch.Tensor, voxel_dir: torch.Tensor, temb: torch.Tensor = None):
        """

        Args:
            features (torch.Tensor): tensor of shape (batch_size, num_images, feature_dim, grid_dim, grid_dim, grid_dim)
            ray_diff (torch.Tensor): tensor of shape (batch_size, num_images, 3, grid_dim, grid_dim, grid_dim)
            mask (torch.Tensor): tensor of shape (batch_size, num_images, 1, grid_dim, grid_dim, grid_dim)
            temb (torch.Tensor) tensor of shape (batch_size, num_images, temb_dim)
        """
        num_images = features.shape[1]

        # add ray encoding and depth
        ray_depth_enc = torch.cat([voxel_dir, voxel_depth], dim=2)
        if torch.isinf(voxel_dir).any():
            print('voxel_dir INF')
        if torch.isinf(voxel_depth).any():
            print('voxel_depth INF')
        if torch.isnan(ray_depth_enc).any():
            print('add ray encoding and depth: 0')
        if torch.isinf(ray_depth_enc).any():
            print('add ray encoding and depth: 0.5')
        ray_depth_enc = collapse_batch(ray_depth_enc)
        x = ray_depth_enc.clone()
        if torch.isnan(ray_depth_enc).any():
            print('add ray encoding and depth: 1')
        if torch.isinf(ray_depth_enc).any():
            print('add ray encoding and depth: 1.5')
        ray_depth_enc = self.ray_depth_encoder(ray_depth_enc)
        if torch.isnan(ray_depth_enc).any():
            print('add ray encoding and depth: 2')
            print('before:',x)
            print('after:',ray_depth_enc)
        ray_depth_enc = expand_batch(ray_depth_enc, num_images)
        if torch.isnan(ray_depth_enc).any():
            print('add ray encoding and depth: 3')
        features = features + ray_depth_enc
        if torch.isnan(features).any():
            print('add ray encoding and depth: nan in features')
            print('features nan:',torch.isnan(features).any())
            print('voxel_dir nan:',torch.isnan(voxel_dir).any())
            print('voxel_depth nan:',torch.isnan(voxel_depth).any())
            

        # add temb encoding
        if self.use_temb:
            temb_enc = self.temb_encoder(temb)
            temb_enc = temb_enc.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            temb_enc = temb_enc.repeat(1, 1, 1, *features.shape[3:]).contiguous()
            features = features + temb_enc

        if self.anti_alias_pooling:
            raise NotImplementedError()
        else:
            weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)\
            
        if torch.isnan(weight).any():
            print('nan in weight')
            

        # compute mean and variance across different views for each voxel (== same as aggregate_cost_volume(agg_fn="mean"))
        mean, var = fused_mean_variance(
            features, weight
        )  # (batch_size, 1, feature_dim, grid_dim, grid_dim, grid_dim)
        globalfeat = torch.cat([mean, var], dim=2)  # (batch_size, 1, 2*feature_dim, grid_dim, grid_dim, grid_dim)

        # combine each voxel with the globalfeat across all views
        # (batch_size, num_images, 3*feature_dim, grid_dim, grid_dim, grid_dim)
        x = torch.cat([globalfeat.expand(-1, num_images, -1, -1, -1, -1), features], dim=2)

        # encode base_fc: shared part of feature/weight encoding
        x = collapse_batch(x)  # (batch_size * num_images, 3*feature_dim, grid_dim, grid_dim, grid_dim)
        weight = collapse_batch(weight)
        mask = collapse_batch(mask)
        x = self.base_fc(x)

        # get averaging weights, final features
        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[1] - 1, 1], dim=1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask

        # compute weighted average
        vis = expand_batch(vis, num_images)
        x = expand_batch(x, num_images)
        weight = vis / (torch.sum(vis, dim=1, keepdim=True) + 1e-8)
        mean, var = fused_mean_variance(x, weight)
       

        # combine (mean, var, weight) and let a final custom layer transform it into the feature grid
        # (batch_size, 2*feature_dim + 1, grid_dim, grid_dim, grid_dim)
        globalfeat = torch.cat([mean.squeeze(1), var.squeeze(1), weight.mean(dim=1)], dim=1)
        globalfeat = self.statistics_out(globalfeat)
        if torch.isnan(globalfeat).any():
            print('nan in globalfeat')
            print('nan in mean:',torch.isnan(mean).any())
            print('nan in var:',torch.isnan(var).any())
            

        return globalfeat



def unproject_features(features, k_matrices, rt_matrices, bbox, latent_dim, grid_dim=48):
    device = features.device
    k_matrices = k_matrices.to(device)
    rt_matrices = rt_matrices.to(device)
    bbox = bbox.to(device)
    batch_size, n_views, c,h,w = features.shape
    volume_batch = build_feature_volume(batch_size, grid_dim, n_views*c, device)   #B*V*V*V*NC 
    volume_std = build_feature_volume(batch_size, grid_dim, c, device)
    grid_volume_batch = build_grid(batch_size, grid_dim, device)    #B*V^3*3
     #B*V^3*3

    v2vnet = V2VNet(c,latent_dim).to(device)
    visnet = VisNet(n_views*c+c,n_views).to(device)

    # TODO: speed up this this loop
    for batch_i in range(batch_size):
        coord_volume = volume_std[batch_i]
        world_volume_batch = gridV2worldV(grid_volume_batch,bbox[batch_i])
        volume_batch_to_aggregate = torch.zeros_like(coord_volume,device = device).unsqueeze(0).repeat(n_views,1,1,1,1)  #N*V*V*V*C

        for view_i in range(n_views):
            feature_map = features[batch_i, view_i]
            #feature_map = feature_map.unsqueeze(0)

            pixel_coord_proj = world2pic(
                k_matrices[batch_i, view_i], rt_matrices[batch_i, view_i], world_volume_batch[batch_i],
            )

          
            invalid_mask1 = pixel_coord_proj[:, 2] <= 0.0  # depth must be larger than 0.0
            

            pixel_coord_proj[pixel_coord_proj[:, 2] == 0.0, 2] = 1.0  # not to divide by zero
            pixel_coord_proj = homogeneous_to_euclidean(pixel_coord_proj)

            invalid_mask2 = pixel_coord_proj[:, 0] < 0.0
            invalid_mask3 = pixel_coord_proj[:, 0] >= h
            invalid_mask4 = pixel_coord_proj[:, 1] < 0.0
            invalid_mask5 = pixel_coord_proj[:, 1] >= w

            #img = np.zeros((384,512,3))
            # img = img.permute(1,2,0).cpu().numpy()
            # pixel = pixel_coord_proj.cpu().numpy()
            # for i in range(pixel.shape[0]):
            #     if 0 < 4*pixel[i,1] <384 and 0 < 4*pixel[i,0] < 512:
            #         img[int(4*pixel[i,1]),int(4*pixel[i,0])] = [0,0,255]
            # cv2.imshow('feature grid',img)
            # cv2.waitKey(0)
            # quit()


            # transform to [-1.0, 1.0] range

            pixel_coord_proj_transformed = torch.zeros_like(pixel_coord_proj)
            pixel_coord_proj_transformed[:, 0] = 2 * (pixel_coord_proj[:, 0] / feature_map.shape[1] - 0.5)
            pixel_coord_proj_transformed[:, 1] = 2 * (pixel_coord_proj[:, 1] / feature_map.shape[0] - 0.5)
            pixel_coord_proj = pixel_coord_proj_transformed
            # prepare to F.grid_sample
            


            pixel_coord_proj = pixel_coord_proj.unsqueeze(1).unsqueeze(0)


            feature_map = feature_map.unsqueeze(0)
            
            current_volume =F.grid_sample(feature_map, pixel_coord_proj, align_corners=True)


            # zero out non-valid points
            current_volume = current_volume.reshape(-1,grid_dim**3)
            current_volume[:, invalid_mask1] = 0.0
            current_volume[:, invalid_mask2] = 0.0
            current_volume[:, invalid_mask3] = 0.0
            current_volume[:, invalid_mask4] = 0.0
            current_volume[:, invalid_mask5] = 0.0

            # reshape back to volume
            current_volume = current_volume.reshape(c,grid_dim,grid_dim,grid_dim)
            current_volume = current_volume.permute(1,2,3,0)
            # collect
            volume_batch_to_aggregate[view_i] = current_volume

        
        # if volume_aggregation_method.startswith('conf'):
        #     volume_batch[batch_i] = (volume_batch_to_aggregate * vol_confidences[batch_i].view(n_views, n_joints, 1, 1, 1)).sum(0)
        # elif volume_aggregation_method == 'sum':
        #     volume_batch[batch_i] = volume_batch_to_aggregate.sum(0)
        # elif volume_aggregation_method == 'max':
        #     volume_batch[batch_i] = volume_batch_to_aggregate.max(0)[0]
        # elif volume_aggregation_method == 'softmax':
        #     volume_batch_to_aggregate_softmin = volume_batch_to_aggregate.clone()
        #     volume_batch_to_aggregate_softmin = volume_batch_to_aggregate_softmin.view(n_views, -1)
        #     volume_batch_to_aggregate_softmin = torch.nn.functional.softmax(volume_batch_to_aggregate_softmin, dim=0)
        #     volume_batch_to_aggregate_softmin = volume_batch_to_aggregate_softmin.view(n_views, n_joints, *volume_shape)

        #     volume_batch[batch_i] = (volume_batch_to_aggregate * volume_batch_to_aggregate_softmin).sum(0)
        # else:
        #     raise ValueError("Unknown volume_aggregation_method: {}".format(volume_aggregation_method))

        # agregate resulting volume
        volume_batch[batch_i] = volume_batch_to_aggregate.permute(1,2,3,0,4).reshape(grid_dim,grid_dim,grid_dim,n_views*c)
        volume_std[batch_i] = volume_batch_to_aggregate.std(0)
    
    input_batch = torch.cat((volume_batch,volume_std),dim=4)
    input_batch = input_batch.permute(0,4,1,2,3)   # input shape:B,N*C,V,V,V
    vision = visnet(input_batch).permute(0,2,3,4,1).unsqueeze(5)      # output shape:B,V,V,V,4
    volume_batch = volume_batch.reshape(batch_size,grid_dim,grid_dim,grid_dim,n_views,c)
    input_volume_batch = (volume_batch * vision).sum(4)   #B,V,V,V,C
    input_volume_batch = input_volume_batch.permute(0,4,1,2,3)
    
    output_batch = v2vnet(input_volume_batch)

    return output_batch


def build_grid(batch_size, grid_dim, dev):
    x = torch.arange(grid_dim) 
    y = torch.arange(grid_dim)
    z = torch.arange(grid_dim)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    grid_xyz = torch.cat(
        [
            grid_x.reshape(grid_dim, grid_dim, grid_dim, 1),
            grid_y.reshape(grid_dim, grid_dim, grid_dim, 1),
            grid_z.reshape(grid_dim, grid_dim, grid_dim, 1),
        ],
        dim=3,
    )

    grid_xyz = grid_xyz.reshape(grid_dim * grid_dim * grid_dim, 3).contiguous()
    grid_volume = grid_xyz.unsqueeze(0).repeat(batch_size,1,1).float()
    grid_volume = grid_volume.to(dev)
    return grid_volume

def build_feature_volume(batch_size, grid_dim, feature_dim, dev):
    feature_volume = torch.zeros(batch_size, grid_dim, grid_dim, grid_dim, feature_dim).to(dev)
    return feature_volume

def gridV2worldV(grid_volume,bbox,V):
    device = grid_volume.device
    origin = bbox[0]
    #size_x, size_y, size_z = bbox[1]
    scale_matrix = torch.diag(bbox[1]/2).to(device)
    grid_volume = (grid_volume-torch.tensor([V-1,V-1,V-1]).to(device)/2)/((V-1)/2)
    world_volume = origin + torch.matmul(grid_volume,scale_matrix)
    return world_volume

def world2grid(world_coords,bbox,V):
    origin = bbox[0]
    size_x, size_y, size_z = bbox[1]
    grid_coords = torch.zeros_like(world_coords).to(world_coords.device)
    scale_matrix = torch.diag(2/bbox[1]).to(world_coords.device)
    grid_coords = torch.matmul((world_coords - origin),scale_matrix)
    grid_coords = grid_coords*(V/2) + torch.tensor([V,V,V],device = world_coords.device)/2 - 0.5
    return grid_coords

def grid2world(grid_coords,bbox,V):
    origin = bbox[0]
    world_coords = torch.zeros_like(grid_coords).to(grid_coords.device)
    scale_matrix = torch.diag(bbox[1]/2).to(grid_coords.device)
    grid_coords = (grid_coords+0.5-torch.tensor([V,V,V],device = grid_coords.device)/2)/(V/2)
    world_coords = origin + torch.matmul(grid_coords, scale_matrix)
    return world_coords

def euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous

    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")

def world2pic(k_matrix, rt_matrix, world_coords):
    world_coords = euclidean_to_homogeneous(world_coords)
    XYZ = (rt_matrix @ world_coords.T).T[:, :3]
    Z = XYZ[:, 2:]
    XYZ = XYZ / XYZ[:, 2:]
    xyz = (k_matrix @ XYZ.T).T
    return xyz
