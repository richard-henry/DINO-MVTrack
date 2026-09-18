import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import utils.basic
from utils.basic import print_stats
import utils.samp
import utils.misc
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from nets.build_cost_volume import *
from track3d.utils.deterministic_sampling import sample_2d, validate_sampling_policy
from track3d.models.encoders import FAMILIES, FrozenFeatureCache, extract_patch_tokens
from track3d.utils.search_size import validate_search_size, initialize_search_size, adapt_search_size
from track3d.utils.correlation_geometry import validate_correlation_geometry, query_aligned_grid, patch_center_intrinsics, patch_center_grid
from track3d.models.query_templates import validate_query_policy, per_view_templates
from track3d.models.view_weights import validate_view_weight_policy, fnet_view_weights
from track3d.models.score_fusion import validate_score_fusion, fuse_scores
from track3d.models.source_evidence import source_templates, sample_source_evidence
from track3d.models.coordinate_update import validate_coordinate_update, center_evidence_update
import nets.camera as camera
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
import cv2
import math
from transformers import AutoImageProcessor, AutoModel, AutoConfig
import time
import torch.utils.checkpoint as checkpoint # 引入库


def visualize_feature_map_pca_global(
    feature_map,  # (B, S, V, C, H8, W8)
    images,       # (B, S, V, 3, H, W), 已归一化到[-1,1]
    b_idx=0,
    v_idx=0,
    pca_dim=3,
    save_dir='vis_global_pca',
):
    os.makedirs(save_dir, exist_ok=True)

    fmap = feature_map[b_idx, :, v_idx].detach().cpu().numpy()  # (S,C,H8,W8)
    imgs = images[b_idx, :, v_idx].detach().cpu().numpy()       # (S,3,H,W)
    S, C, H8, W8 = fmap.shape
    _, _, H, W = imgs.shape

    # ---------- Step 1: reshape 所有帧特征为 (S*H8*W8, C) ----------
    all_feats = fmap.transpose(0, 2, 3, 1).reshape(-1, C)  # (S*H8*W8, C)

    # ---------- Step 2: 拟合全局 PCA ----------
    pca = PCA(n_components=pca_dim)
    all_pcs = pca.fit_transform(all_feats)  # (S*H8*W8, pca_dim)

    # ---------- Step 3: 还原每帧 ----------
    pcs_per_frame = []
    ptr = 0
    for s in range(S):
        cur = all_pcs[ptr:ptr + H8*W8]
        ptr += H8*W8
        cur = cur.reshape(H8, W8, pca_dim)
        pcs_per_frame.append(cur)

    # ---------- Step 4: 归一化到 [0,1] ----------
    pcs_min, pcs_max = all_pcs.min(), all_pcs.max()
    pcs_per_frame = [
        (np.clip(frame, pcs_min, pcs_max) - pcs_min) / (pcs_max - pcs_min + 1e-8)
        for frame in pcs_per_frame
    ]

    # ---------- Step 5: 插值 & 可视化 ----------
    results = []
    for s in range(S):
        fmap_vis = pcs_per_frame[s]  # (H8,W8,3 or 1)
        # Resize to original image size
        fmap_resized = cv2.resize(fmap_vis, (W, H), interpolation=cv2.INTER_CUBIC)

        if pca_dim == 1:
            # 灰度热力图
            fmap_gray = np.uint8(255 * fmap_resized)
            heatmap = cv2.applyColorMap(fmap_gray, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        elif pca_dim == 3:
            heatmap = np.uint8(255 * fmap_resized)
        else:
            raise ValueError("pca_dim must be 1 or 3")

        # ---------- Step 6: 原图反归一化 ----------
        img = imgs[s]
        img = (img + 1) / 2.0  # [-1,1] -> [0,1]
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))  # (H,W,3)

        # ---------- Step 7: 叠加 ----------
        overlay = cv2.addWeighted(img, 0.2, heatmap, 0.8, 0)
        save_path = os.path.join(save_dir, f"b{b_idx}_v{v_idx}_s{s:02d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        results.append(overlay)

    print(f"✅ Saved {S} global PCA visualizations to {save_dir}")
    
    return results

def visualize_point_correlation(point_coords, point_feature, original_image, feature_map, target_image, alpha=0.5, cmap='jet'):
    """
    Args:
        point_coords: Tensor of shape (2,), pixel coordinates (x, y) on original image
        point_feature: Tensor of shape (C,)
        feature_map: Tensor of shape (C, H_feat, W_feat)
        original_image: Tensor of shape (3, H, W)
        target_image: Tensor of shape (3, H, W)
        alpha: blending factor
        cmap: colormap
    """
    assert feature_map.dim() == 3 and point_feature.dim() == 1
    C, H_feat, W_feat = feature_map.shape
    _, H_img, W_img = target_image.shape[-3:] # 获取目标图像的原始尺寸

    # --- 关键修改：动态计算缩放比例 ---
    scale_h = H_img / H_feat
    scale_w = W_img / W_feat

    # --- Step 1: correlation map ---
    # 点乘计算相关性： (C, H_feat, W_feat) 和 (C,) 做点积 → (H_feat, W_feat)
    corr_map = torch.einsum('chw,c->hw', feature_map, point_feature)
    
    # 归一化便于 Softmax 和可视化
    corr_map = corr_map - corr_map.min()
    corr_map = corr_map / (corr_map.max() + 1e-8)

    # 1.1 寻找 Max 坐标 (在特征图尺度)
    max_idx = corr_map.argmax()
    max_y_feat = max_idx // W_feat
    max_x_feat = max_idx % W_feat

    # 1.2 转换为原图坐标 (应用动态 Scale)
    # 加 0.5 是为了对应到像素中心，不仅更准，而且防止边缘对齐问题
    max_y = (max_y_feat + 0.5) * scale_h
    max_x = (max_x_feat + 0.5) * scale_w

    # 1.3 Soft-Argmax (期望坐标)
    temperature = 0.01
    corr_flat = corr_map.view(-1)
    prob = torch.softmax(corr_flat / temperature, dim=0)

    # 生成网格 (特征图尺度)
    ys, xs = torch.meshgrid(torch.arange(H_feat, device=corr_map.device),
                            torch.arange(W_feat, device=corr_map.device),
                            indexing='ij')
    xs, ys = xs.reshape(-1).float(), ys.reshape(-1).float()

    # 加权平均 (得到特征图尺度上的期望坐标)
    exp_x_feat = (prob * xs).sum()
    exp_y_feat = (prob * ys).sum()

    # 转换为原图坐标
    exp_x = (exp_x_feat + 0.5) * scale_w
    exp_y = (exp_y_feat + 0.5) * scale_h

    # --- Step 2: resize correlation map to image size ---
    # 插值回原图尺寸用于显示
    corr_map_up = F.interpolate(corr_map.unsqueeze(0).unsqueeze(0), size=(H_img, W_img), mode='bilinear', align_corners=False)
    corr_map_up = corr_map_up.squeeze().cpu().numpy()

    # --- Step 3: convert to heatmap ---
    cmap_fn = plt.get_cmap(cmap)
    heatmap = cmap_fn(corr_map_up)[:, :, :3]  # RGBA → RGB
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1)  # (3, H, W)
    
    # --- Step 4: overlay on image ---
    img = target_image.clone().detach().cpu()
    # 归一化处理
    if img.min() < 0:
        img = (img + 1) / 2.0
        # 对 original_image 也做同样处理以便显示
        original_image = (original_image + 1) / 2.0
    elif img.max() > 1.0:
        img = img / 255.0
        original_image = original_image / 255.0
        
    # 简单的 clamp 防止数值溢出
    img = torch.clamp(img, 0, 1)
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.detach().cpu()
        if original_image.min() < 0: original_image = (original_image + 1) / 2.0
        elif original_image.max() > 1.0: original_image = original_image / 255.0
        original_image = torch.clamp(original_image, 0, 1)

    overlay = (1 - alpha) * img + alpha * heatmap

    # --- Step 5: visualize ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # 图1: Query Image + Query Point
    ax[0].imshow(original_image.permute(1, 2, 0))
    ax[0].scatter([point_coords[0].item()], [point_coords[1].item()], c='r', s=40, label='Query')
    ax[0].set_title('Original Image (Query)')
    ax[0].legend()

    # 图2: Heatmap + Predicted Point
    ax[1].imshow(corr_map_up, cmap=cmap)
    # 注意: matplotlib 的 scatter 接受的是 (x, y)，不需要转为整数，float 更准
    ax[1].scatter([max_x.item()], [max_y.item()], c='r', marker='x', s=60, label='Max')
    ax[1].scatter([exp_x.item()], [exp_y.item()], c='b', marker='+', s=60, label='SoftArg')
    ax[1].set_title('Correlation Heatmap')
    ax[1].legend()

    # 图3: Overlay + Predicted Point
    ax[2].imshow(overlay.permute(1, 2, 0))
    ax[2].scatter([max_x.item()], [max_y.item()], c='r', marker='x', s=60)
    ax[2].scatter([exp_x.item()], [exp_y.item()], c='b', marker='+', s=60)
    ax[2].set_title('Target Image Overlay')

    plt.tight_layout()
    plt.show()

    return corr_map_up, overlay


def visualize_voxel_correlation_heatmap(
    volume_2d_correlation, volume_coords, images,itr,
    b=0, s=5, n=2, v=1, 
    threshold=0.05,
    dilation_size=3,
    cmap='jet'
):
    """
    在视角 v 的图像上显示体素可视化（不叠加，不修改原图）。
    - 先对体素correlation归一化
    - 显示体素颜色
    - mask 膨胀让体素更明显
    - 同时显示原图、mask、可视化结果

    Args:
        threshold: 体素显示阈值
        dilation_size: mask膨胀大小（奇数 3/5/7）
    """

    _, _, _, _, H, W = images.shape

    # ===============================
    # 1️⃣ 取出数据
    # ===============================
    volume = volume_2d_correlation[b, s, n]     # (g,g,g)
    coords = volume_coords[b, s, v, n]          # (g,g,g,2)
    img = images[b, s, v].permute(1, 2, 0).cpu().numpy()
    img = np.clip(img / (img.max() + 1e-6), 0, 1)   # 归一化显示，不修改原始

    query_point = volume_coords[b,0,v,n].reshape(-1,2).mean(0)


    # 展平体素
    coords_flat = coords.reshape(-1, 2).cpu().numpy()
    volume_flat = volume.reshape(-1).cpu().numpy()

    # ===============================
    # 2️⃣ 对 correlation 做归一化（重要）
    # ===============================
    vol_min = volume_flat.min()
    vol_max = volume_flat.max()
    if vol_max - vol_min < 1e-6:
        volume_norm = np.zeros_like(volume_flat)
    else:
        volume_norm = (volume_flat - vol_min) / (vol_max - vol_min)

    # ===============================
    # 3️⃣ 投影为 max-map（取每个像素的最大值）
    # ===============================
    maxmap = np.zeros((H, W), dtype=np.float32)

    for i in range(len(volume_norm)):
        u, v_ = coords_flat[i]
        ui, vi = int(round(u)), int(round(v_))
        if 0 <= ui < W and 0 <= vi < H:
            val = volume_norm[i]
            if val > maxmap[vi, ui]:
                maxmap[vi, ui] = val

    # ===============================
    # 4️⃣ 阈值 mask + 膨胀
    # ===============================
    mask = maxmap > threshold
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel)

    # ===============================
    # 5️⃣ 用 colormap 显示体素颜色，而不是蓝色
    # ===============================
    heat_color = cv2.applyColorMap(
        (maxmap * 255).astype(np.uint8),
        getattr(cv2, f'COLORMAP_{cmap.upper()}')
        if hasattr(cv2, f'COLORMAP_{cmap.upper()}') else cv2.COLORMAP_JET
    )
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB) / 255.0

    # 可视化图（不修改原图）
    img_vis = img.copy()
    img_vis[mask_dilated == 1] = heat_color[mask_dilated == 1]

    radius = 3                  # 半径，可以调大一点
    color = (1, 0, 0)           # 红色，注意 img_vis 是 0~1 float
    thickness = -1              # 填充圆

    cv2.circle(img_vis, (int(query_point[0]), int(query_point[1])), radius, color, thickness)

    img_to_save = (img_vis * 255).astype(np.uint8)
    cv2.imwrite("img_vis_%d.png"%(itr), cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))

    mask_to_save = (mask * 255).astype(np.uint8)
    cv2.imwrite("mask_%d.png"%(itr), mask_to_save)

    # ===============================
    # 6️⃣ 显示：原图 / mask / 结果
    # ===============================
    # plt.figure(figsize=(14, 4))

    # plt.subplot(1, 3, 1)
    # plt.imshow(img)
    # plt.title("Original Image")
    # plt.axis("off")

    # plt.subplot(1, 3, 2)
    # plt.imshow(mask_dilated, cmap='gray')
    # plt.title("Mask (Dilated)")
    # plt.axis("off")

    # plt.subplot(1, 3, 3)
    # plt.imshow(img_vis)
    # plt.title("Voxel Visualization")
    # plt.axis("off")

    # plt.show()

    return img_vis, mask_dilated, maxmap

def sample_scores_from_heatmap(heatmap, grid_coords, sampling_policy='native'):
    """
    从 N 张热力图中，分别使用 N 个对应的 Grid 进行采样。
    
    Args:
        heatmap: (B, S, V, N, H, W)  <-- 上采样后的高分辨率热力图 (单通道)
        grid_coords: (B, S, V, N, G^3, 2) <-- 对应于 H, W 的像素坐标
        
    Returns:
        scores: (B, S, N, V, G^3)
    """
    B, S, V, N, H, W = heatmap.shape
    _, _, _, _, G3, _ = grid_coords.shape
    
    # 1. 维度重组：将 (B,S,V,N) 合并为一个大的 Batch 维度
    # Heatmap: (Batch_All, 1, H, W)
    heatmap_flat = heatmap.reshape(B*S*V*N, 1, H, W)
    
    # Grid: (Batch_All, G^3, 1, 2)
    # 增加维度 1 是为了匹配 grid_sample 的 (N, H_out, W_out, 2) 格式
    grid_flat = grid_coords.reshape(B*S*V*N, G3, 1, 2)
    
    # 2. 坐标归一化 (Pixel -> [-1, 1])
    # grid_sample 需要 [-1, 1] 区间
    # 关键：这里的 H, W 是上采样后的尺寸！
    grid_norm = torch.zeros_like(grid_flat)
    grid_norm[..., 0] = 2.0 * grid_flat[..., 0] / (W - 1) - 1.0 # x
    grid_norm[..., 1] = 2.0 * grid_flat[..., 1] / (H - 1) - 1.0 # y
    
    # 3. 采样 (显存极低，因为通道数是 1)
    # Output: (Batch_All, 1, G^3, 1)
    sampled = sample_2d(heatmap_flat, grid_norm, align_corners=True, mode='bilinear', sampling_policy=sampling_policy)
    
    # 4. 还原维度并对齐
    # (B*S*V*N, 1, G^3, 1) -> (B, S, V, N, G^3) -> (B, S, N, V, G^3)
    scores = sampled.view(B, S, V, N, G3).permute(0, 1, 3, 2, 4)
    
    return scores

class Conv1dPad(nn.Module):
    """
    nn.Conv1d with auto-computed padding ("same" padding)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(Conv1dPad, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        return net
    
class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, use_norm, use_do, is_first_block=False):
        super(ResidualBlock1d, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.stride = 1
        self.is_first_block = is_first_block
        self.use_norm = use_norm
        self.use_do = use_do

        self.norm1 = nn.InstanceNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = Conv1dPad(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        self.norm2 = nn.InstanceNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = Conv1dPad(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)

    def forward(self, x):
        
        identity = x
        
        out = x
        if not self.is_first_block:
            if self.use_norm:
                out = self.norm1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        if self.use_norm:
            out = self.norm2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
            
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        out += identity
        return out

def sequence_loss(flow_preds, flow_gt, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    B, S, N, D = flow_gt.shape
    assert(D==3)
    #B, S1, N = vis.shape
    #B, S2, N = valids.shape
    #assert(S==S1)
    #assert(S==S2)
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        flow_pred = flow_preds[i]
        i_loss = (flow_pred - flow_gt).abs() # B,S,N,3
        i_loss = torch.mean(i_loss, dim=-1) # B,S,N
        flow_loss += i_weight * torch.sum(i_loss)
    flow_loss = flow_loss/n_predictions
    return flow_loss

def vis_loss(vis, visg):
    bce = nn.BCELoss()
    return bce(vis, visg)

def compute_L_prior(W_raw, delta, V):
    """
    计算 ViewAttentionMLP 的强引导先验损失 L_prior。
    惩罚 W_raw_self 不足以显著高于 W_raw_max_other 的情况。
    
    Args:
        W_raw: (B, 1, N, Vq, Vt) 或 (M_view, 1) 形状的原始注意力分数
        delta: 安全边距 (例如 0.5 到 1.0)，用于定义所需的 W_self 差距
        V: 视角数量
        lambda_prior: 损失权重系数 (用于返回 L_prior * lambda_prior)
        
    Returns:
        L_prior: 标量，当前迭代的加权先验损失
    """
    
    # 确保 W_raw 形状是 (B*1*N, Vq, Vt)
    if W_raw.dim() == 2:
        # 如果是 (M_view, 1) 展平形状，需要重塑
        M_view = W_raw.shape[0]
        N_pts = M_view // (V * V)
        B = 1
        W_raw_flat = W_raw.reshape(B, 1, N_pts, V, V) # (B, 1, N, V, V)
    else:
        # 假设输入已经是 (B, 1, N, V, V) 形状
        W_raw_flat = W_raw
        
    B, _, N, Vq, Vt = W_raw_flat.shape
    
    # 将形状展平为 (B*N, Vq, Vt)，忽略时间/批次维度
    W_raw_reduced = W_raw_flat.reshape(-1, Vq, Vt) 
    device = W_raw.device

    # 1. W_raw_self：提取同视角权重 (对角线)
    diag_mask = torch.eye(Vq, Vt, device=device).bool() 
    # W_self 形状为 (B*N, V)
    W_self = W_raw_reduced[:, diag_mask].reshape(-1, Vq) 

    # 2. W_raw_other：提取异视角权重
    other_mask = ~diag_mask 
    # W_other_scores 形状为 (B*N, V, V-1)
    W_other_scores = W_raw_reduced[:, other_mask].reshape(-1, Vq, Vt - 1) 
    
    # 3. W_raw_max_other：异视角中的最大权重
    W_max_other, _ = torch.max(W_other_scores, dim=-1) # (B*N, Vq)

    # 4. 计算 L_prior：惩罚 W_self 小于 W_max_other + delta 的情况
    # 注意：我们直接操作原始分数，因此 delta 应该相对较大 (例如 0.5 到 1.0)
    L_prior_raw = torch.relu(W_max_other - W_self + delta)

    # 5. L_prior_i：求均值，并乘以 lambda_prior
    L_prior_i = L_prior_raw.mean()

    return L_prior_i


def correlation(query_feat, target_feat):
    B1, S1, N1, C1 = query_feat.shape
    B2, S2, N2, num, C2 = target_feat.shape
    assert B1 == B2 and N1 == N2 and S1 == S2 and C1 == C2
    
    corrs = torch.einsum('bsnc,bsnmc->bsnm', query_feat, target_feat)
    corrs = corrs / torch.sqrt(torch.tensor(C2, dtype=torch.float32, device=corrs.device))
    
    return corrs

def get_3d_embedding(xyz, C, cat_coords=True):
    B, N, D = xyz.shape
    assert D == 3

    x = xyz[:, :, 0:1]
    y = xyz[:, :, 1:2]
    z = xyz[:, :, 2:3]
    div_term = (
            torch.arange(0, C, 2, device=xyz.device, dtype=torch.float32) * (1000.0 / C)
    ).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_z = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)

    pe = torch.cat([pe_x, pe_y, pe_z], dim=2)  # B, N, C*3
    if cat_coords:
        pe = torch.cat([pe, xyz], dim=2)  # B, N, C*3+3
    return pe



class ResidualBlock2d(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock2d, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, padding_mode='zeros')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
    
    
class SpaEncoder(nn.Module):
    def __init__(
        self, input_dim=3, output_dim=128, stride=8, norm_fn="batch", dropout=0.0,
        Embed3D=False
    ):
        super(SpaEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn
        self.in_planes = 64

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim * 2)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.shallow = False
        if self.shallow:
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(128 + 96 + 64, output_dim, kernel_size=1)
        else:
            if Embed3D:
                self.conv_fuse = nn.Conv2d(64+63, 
                                           self.in_planes, kernel_size=3, padding=1)
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.layer4 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(
                128 + 128 + 96 + 64,
                output_dim * 2,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
            )
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                                 nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock2d(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock2d(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, feat_PE=None):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        if self.shallow:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            a = F.interpolate(
                a,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            b = F.interpolate(
                b,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            c = F.interpolate(
                c,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            x = self.conv2(torch.cat([a, b, c], dim=1))
        else:
            if feat_PE is not None:
                x = self.conv_fuse(torch.cat([x, feat_PE], dim=1))
            
            if self.training: # 只在训练时开启，推理时不需要
                a = checkpoint.checkpoint(self.layer1, x, use_reentrant=False)
                b = checkpoint.checkpoint(self.layer2, a, use_reentrant=False)
                c = checkpoint.checkpoint(self.layer3, b, use_reentrant=False)
                d = checkpoint.checkpoint(self.layer4, c, use_reentrant=False)
            else:
                a = self.layer1(x)
                b = self.layer2(a)
                c = self.layer3(b)
                d = self.layer4(c)


            a_up = F.interpolate(a,(H // self.stride, W // self.stride),mode="bilinear",align_corners=True,)
            b_up = F.interpolate(b,(H // self.stride, W // self.stride),mode="bilinear",align_corners=True,)
            c_up = F.interpolate(c,(H // self.stride, W // self.stride),mode="bilinear",align_corners=True,)
            d_up = F.interpolate(d,(H // self.stride, W // self.stride),mode="bilinear",align_corners=True,)


            x = self.conv2(torch.cat([a_up, b_up, c_up, d_up], dim=1))
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        return x

class DeltaBlock(nn.Module):
    def __init__(self, latent_dim=96, posenc_dim=64, grid_dim = 10, corr_levels = 4):
        super(DeltaBlock, self).__init__()
        
        self.kitchen_dim = posenc_dim*3 + 3 + grid_dim**3

        self.output_dim = 3 + 1 
        
        in_channels = self.kitchen_dim
        base_filters = 256
        self.n_block = 8
        self.kernel_size = 3
        self.groups = 1
        self.use_norm = True
        self.use_do = False

        self.increasefilter_gap = 2 

        self.first_block_conv = Conv1dPad(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_norm = nn.InstanceNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):

            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False

            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = ResidualBlock1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride=1, 
                groups=self.groups, 
                use_norm=self.use_norm, 
                use_do=self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        self.final_norm = nn.InstanceNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense1 = nn.Linear(out_channels, self.output_dim)
        
            
    def forward(self, x):
        B, S, D = x.shape
        assert(D==self.kitchen_dim)
        # conv1d wants channels in the middle
        out = x.permute(0,2,1)
        out = self.first_block_conv(out)
        out = self.first_block_relu(out)
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            out = net(out)
        out = self.final_relu(out)
        out1 = out.permute(0,2,1)
        delta = self.dense1(out1)

        return delta
    
class VisibilityNet(nn.Module):
    def __init__(self, L, hidden_dims=[64, 32, 16]):
        """
        Args:
            L: 分辨率层数
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        
        layers = []
        input_dim = L
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # output layer
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, correlation):
        """
        Args:
            correlation: [B, N, L]
        
        Returns:
            visibility: [B, N, 1]
        """
        B, N, L = correlation.shape
        output = self.mlp(correlation)  # [B, N, 1]
        
        return output
    
class Residual1x1VoxelNet_IN(nn.Module):
    def __init__(self, L, hidden_dim=64, num_blocks=4):
        super().__init__()
        
        # 初始投影
        self.initial = nn.Conv3d(L, hidden_dim, 1)
        
        # 使用InstanceNorm的残差块
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(hidden_dim, hidden_dim, 1),
                nn.InstanceNorm3d(hidden_dim),  # 替换为InstanceNorm
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden_dim, hidden_dim, 1),
                nn.InstanceNorm3d(hidden_dim),  # 替换为InstanceNorm
            ) for _ in range(num_blocks)
        ])
        
        # 输出层
        self.output = nn.Conv3d(hidden_dim, 1, 1)
        
    def forward(self, correlation_volume):
        B, S, N, L, g, _, _ = correlation_volume.shape
        x = correlation_volume.view(B * S * N, L, g, g, g)
        
        x = self.initial(x)
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
        
        x = self.output(x).squeeze(1)
        return x.view(B, S, N, g, g, g)
    
class LightweightConfidenceNet(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        
        # 池化操作
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 简单的线性层
        self.linear = nn.Sequential(
            nn.Linear(2 * L, 32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, correlation_volume):
        B, S, N, L, g, _, _ = correlation_volume.shape
        
        x = correlation_volume.view(B * S * N, L, g, g, g)
        
        # 池化特征
        avg_features = self.avg_pool(x).view(B * S * N, L)
        max_features = self.max_pool(x).view(B * S * N, L)
        
        # 拼接并线性变换
        combined = torch.cat([avg_features, max_features], dim=-1)
        conf = self.linear(combined).view(B, S, N, 1)

        return conf

class GeometricStatsEncoder(nn.Module):
    def __init__(self, output_dim=8):
        super().__init__()
        
        # 统计量维度: 像素极差 (2D) + 像素标准差 (2D) = 4D
        self.encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, output_dim), # 输出 F_stat 维度
        )
        
    def forward(self, volume_coords):
        """
        计算体素块在每个视角下的几何统计指纹。
        
        Args:
            volume_coords: (B, S, V, N, g, g, g, 2)  - 特征图尺度下的像素坐标
            volume_depths: (B, S, V, N, g, g, g, 1)  - 体素中心点的深度
            
        Returns:
            F_stat: (B, S, V, N, D_stat)
        """
        
        B, S, V, N, g, _, _, D_coords = volume_coords.shape
        M = B * S * N * V # 总样本数 (点-时间-视图)
        
        # 将体素展平到 (M, g*g*g, D)
        coords_flat = volume_coords.reshape(M, g*g*g, D_coords)
        # depths_flat = volume_depths.reshape(M, g*g*g, 1)

        # 1. 像素极差 (Pixel Range / Scale) - [2D]
        p_max, _ = coords_flat.max(dim=1) 
        p_min, _ = coords_flat.min(dim=1) 
        range_stat = p_max - p_min # (M, 2)

        # 2. 像素标准差 (Pixel Standard Deviation / Density) - [2D]
        # 注意: std() 在 batch_size=1, N=1, V=1 时可能会出现 nan/0
        std_stat = coords_flat.std(dim=1).nan_to_num(0.0) # (M, 2)
        if torch.isnan(std_stat).any():
            print('nan in std_stat')
            print('coords_flat:',coords_flat)
            quit()
        
        
        # 3. 平均深度 (Mean Depth) - [1D]
        # mean_depth = depths_flat.mean(dim=1) # (M, 1)
        
        # 拼接所有统计量
        stats = torch.cat([range_stat, std_stat], dim=-1) # (M, 4)
        
        # 编码
        F_stat = self.encoder(stats) # (M, output_dim)

        return F_stat.reshape(B, S, V, N, -1) # (B, S, V, N, D_stat)
    

class ViewAttentionMLP(nn.Module):
    """
    动态预测 Vq -> Vt 权重（W_raw），基于相关性质量和几何上下文。
    
    输入: 4 (Corr Metrics) + D_stat + D_scale_embed
    输出: 1 (原始权重 W_raw)
    """
    def __init__(self, D_scale_embed, hidden_dim=64):
        super().__init__()
        
        # 4D 纯相关性特征 + D_scale_embed 尺度特征
        D_attn_in = 2 + D_scale_embed
        
        # 增加深度和宽度以增强表达能力 (64 -> 64 隐藏层)
        self.attn_net = nn.Sequential(
            nn.Linear(D_attn_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1), 
            nn.Tanh()   # 输出原始权重 W_raw
        )
        
    def forward(self, input_features):
        """
        Args:
            input_features: (M_view, 4 + D_stat + D_scale_embed)
        Returns:
            W_raw: (M_view, 1)
        """
        return self.attn_net(input_features)
    


    
class VoxelLevelFusion_Attention(nn.Module):
    def __init__(self, lvl, D_stat, D_scale_embed, scale_embeddinmg):
        super().__init__()
        self.lvl = lvl
        self.D_stat = D_stat
        self.D_scale_embed = D_scale_embed
        
        # 尺度层级编码
        self.scale_embedding = scale_embeddinmg
        
        # 最终的嵌入维度是 lvl * D_scale_embed
        D_total_embed = lvl * D_scale_embed
        D_in = 3 * lvl + D_stat + D_total_embed
    
        # Level Weight Predictor 输入维度: 2*lvl + D_stat + D_total_embed
        self.level_weight_predictor = nn.Sequential(
            nn.Linear(D_in, 128), 
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, lvl),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, corr_stack, F_stat):
        
        B, S, N, V, lvl, g, _, _ = corr_stack.shape 
        M = B * S * N * V 
        
        # 0. 维度对齐
        F_stat_aligned = F_stat.permute(0, 1, 3, 2, 4) # (B, S, N, V, D_stat)

        # 1. 提取 Mean/Max 统计量 (Attention Input Base)
        corr_flat = corr_stack.reshape(B, S, N, V, lvl, g*g*g) 
        corr_means_per_level = corr_flat.mean(dim=-1) 
        corr_maxs_per_level, _ = corr_flat.max(dim=-1)
        corr_stds_per_level = corr_flat.std(dim=-1).nan_to_num(0.0)
        
        # 拼接 Mean 和 Max: (B, S, N, V, 2 * lvl)
        corr_input_feat = torch.cat([corr_means_per_level, corr_maxs_per_level, corr_stds_per_level], dim=-1)
        
        # --- 🌟 尺度嵌入的正确注入 ---
        
        # A. 生成尺度嵌入 (lvl) -> (lvl, D_scale_embed)
        lvl_indices = torch.arange(lvl, device=corr_stack.device).long()
        scale_embed_raw = self.scale_embedding(lvl_indices)
        
        # B. 将 L 个嵌入向量在通道维度上拼接，形成一个超向量
        # (lvl, D_scale_embed) -> (1, lvl * D_scale_embed)
        scale_embed_vector = scale_embed_raw.reshape(1, -1)
        
        # C. 扩展到 M 个样本: (M, lvl * D_scale_embed)
        scale_embed_guide = scale_embed_vector.repeat(M, 1)
        
        # D. 展平特征
        F_stat_flat = F_stat_aligned.reshape(M, self.D_stat) 
        corr_input_feat_flat = corr_input_feat.reshape(M, 3 * lvl)
        
        # 最终输入拼接: (M, 2*lvl + D_stat + lvl*D_scale_embed)
        input_feat_flat = torch.cat([
            corr_input_feat_flat, 
            F_stat_flat,
            scale_embed_guide # 🌟 修正后的嵌入超向量
        ], dim=-1) 
        
        # 3. 预测尺度权重 (Attention Weights)
        # 注意: 如果 D_scale_embed=4, lvl=3, 则 total_embed_dim = 12
        # 我们必须在 __init__ 中修正 level_weight_predictor 的输入维度
        # ... (预测和融合逻辑不变)
        
        scale_weights = self.level_weight_predictor(input_feat_flat)
        
        # 4. 加权求和融合尺度维度
        # 权重形状调整: (M, lvl) -> (M, lvl, 1, 1, 1)
        weights_reshaped = scale_weights.reshape(M, lvl, 1, 1, 1)
        
        # corr_stack 形状展平: (M, lvl, g, g, g)
        # 🚨 注意：corr_stack 需要重塑以确保维度对应
        
        # corr_stack 在拼接前被堆叠为 (B, S, N, V, lvl, g, g, g)
        # 我们需要将其展平为 (M, lvl, g, g, g)
        
        # 确保 M*lvl 的维度对应于 corr_stack 的 (B*S*N*V*lvl)
        corr_stack_flat = corr_stack.reshape(M, lvl, g, g, g) 
        
        # 加权融合: (M, g, g, g) -> (B, S, N, V, g, g, g)
        corr_level_fused = (corr_stack_flat * weights_reshaped).sum(dim=1).reshape(B, S, N, V, g, g, g)
        
        return corr_level_fused, scale_weights.reshape(B, S, N, V, lvl)


class MultiScaleCorrelationFusionWithVisibility_ScaleFirst_Attention(nn.Module):
    def __init__(self, lvl, scale_embeddinmg, hidden_dim=64, D_stat=8, D_scale_embed=4):
        super().__init__()
        self.lvl = lvl
        self.D_stat = D_stat
        
        # 尺度融合模块 (保持不变)
        self.level_fusion_module = VoxelLevelFusion_Attention(lvl, D_stat, D_scale_embed, scale_embeddinmg)

        # 1. 可见性预测器 (回答"是否可见"，用于计算 BCE Loss)
        # 输入维度: 3 (Mean, Max, Std) + D_stat
        self.visibility_predictor = nn.Sequential(
            nn.Linear(2 + D_stat, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            # nn.Sigmoid()
        )

        # 2. 视角质量/权重预测器 (回答"该视角多大程度上值得信任")
        # 核心改进：专门学习如何利用 Variance/Std 来给平坦的分布降权
        self.view_quality_predictor = nn.Sequential(
            nn.Linear(3 + D_stat, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), # 加深一层以学习非线性关系(如方差小->权重极低)
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # 输出 Logit，后续过 Softmax
        )
        
    def forward(self, correlation_volumes, F_stat):
        
        B, S, N, V, g, _, _ = correlation_volumes[0].shape
        M = B * S * N * V

        # 1. --- 尺度融合 (Level Fusion) ---
        corr_stack = torch.stack(correlation_volumes, dim=4) 
        corr_level_fused, scale_weights = self.level_fusion_module(corr_stack, F_stat) 

        # 2. --- 提取能够反映"判别力"的统计特征 ---
        # corr_level_fused: (B, S, N, V, g, g, g)
        corr1 = corr_level_fused.reshape(M, -1)
        
        mean_pool = corr1.mean(dim=-1, keepdim=True) # (M, 1)
        max_pool, _ = corr1.max(dim=-1, keepdim=True) # (M, 1)
        std_pool = corr1.std(dim=-1, keepdim=True).nan_to_num(0.0) # (M, 1)
        
        # 拼接几何统计特征 (F_stat)
        F_stat_flat = F_stat.reshape(M, self.D_stat) 

        vis_input = torch.cat([mean_pool, max_pool, F_stat_flat], dim=-1)
        # 输入特征: (M, 3 + D_stat)
        quality_input = torch.cat([mean_pool, max_pool, std_pool, F_stat_flat], dim=-1)

        # 3. --- 分支 A: 预测可见性 (用于外部 Loss 监督) ---
        # 输出范围 [0, 1]
        visibility_logits = self.visibility_predictor(vis_input).squeeze(-1).reshape(B, S, N, V)

        visibility_prob = torch.sigmoid(visibility_logits)

        # 4. --- 分支 B: 预测融合权重 (用于加权求和) ---
        # 输出为 Logits
        quality_logits = self.view_quality_predictor(quality_input).squeeze(-1).reshape(B, S, N, V)

        # 5. --- 视角融合 (View Fusion) ---
        
        # 策略改进：融合权重 = Softmax(质量 Logits) * 可见性掩码
        # 解释：
        # - view_quality_predictor 会学到：当 Std 很小时，输出极小的 Logit，从而在 Softmax 中获得低权重。
        # - visibility_pred 作为一个硬门控：如果完全被遮挡(vis->0)，即使分布很锐利(比如噪声)，也不应该采纳。
        
        # 步骤 5.1: 计算基础 Attention 权重 (Softmax 保证和为1)
        # 可以在这里除以一个 temperature (如 0.1) 来进一步锐化权重
        attn_weights = F.softmax(quality_logits, dim=-1) 
        
        # 步骤 5.2: 结合可见性进行加权 (Optional but Recommended)
        # 这样既考虑了分布锐度(attn)，也考虑了物理遮挡(vis)
        final_weights = attn_weights * visibility_prob
        
        # 步骤 5.3: 再次归一化，防止 vis 全为 0 导致除零
        final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 使用最终权重进行加权求和
        score_volume = (corr_level_fused * final_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=3)

        return score_volume.unsqueeze(-1), visibility_logits


class MultiScaleCorrelationFusionWithVisibility(nn.Module):
    def __init__(self, lvl, hidden_dim=16, use_attention=True):
        super().__init__()
        self.lvl = lvl
        self.use_attention = use_attention

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # 每层的可见度预测器（共享结构）
        self.visibility_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(lvl)
        ])

        # 可选的多尺度注意力模块
        if use_attention:
            self.level_attention = nn.Sequential(
                nn.Linear(2 * lvl, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, lvl),
                nn.Softmax(dim=-1)
            )

        # 多尺度融合 1×1×1 卷积
        # self.fusion_conv = nn.Conv3d(1, 1, kernel_size=1)

    def forward(self, correlation_volumes):
        """
        Args:
            correlation_volumes: list of lvl tensors, each (B,S,N,V,g,g,g)
        Returns:
            score_volume: (B,S,N,g,g,g)
            visibility_pred: (B,S,N,V)
        """
        B, S, N, V, g, _, _ = correlation_volumes[0].shape

        visibility_levels = []
        fused_per_level = []

        for l in range(self.lvl):
            corr = correlation_volumes[l]  # (B,S,N,V,g,g,g)

            # ---- (1) 可见度特征提取 ----
            corr1 = corr.reshape(B*S*N,V,-1)
            mean_pool = corr1.mean(dim=-1)  # (B*S*N,V)
            max_pool, _ = corr1.max(dim=-1) # (B*S*N,V)
            pooled = torch.stack([mean_pool, max_pool], dim=-1)  # (B*S*N,V,2)

            # ---- (2) 可见度预测 ----
            visibility = self.visibility_predictors[l](pooled).squeeze(-1).reshape(B,S,N,V)  # (B*S*N,V)
            visibility_levels.append(visibility)

            # ---- (3) 按可见度加权融合 ----
            vis_norm = visibility / (visibility.sum(dim=-1, keepdim=True) + 1e-8)
            fused_corr = (corr * vis_norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=3)  # (B,S,N,g,g,g)
            fused_per_level.append(fused_corr)

        # ---- (4) 多尺度融合 ----
        fused_stack = torch.stack(fused_per_level, dim=3)  # (B,S,N,lvl,g,g,g)

        if self.use_attention:
            lvl_mean = fused_stack.reshape(B,S,N,self.lvl,-1).mean(dim=-1)   # (B,S,N,lvl)
            lvl_max = fused_stack.reshape(B, S, N, self.lvl, -1).max(dim=-1).values
            lvl_feat = torch.cat([lvl_mean, lvl_max], dim=-1)
            attn = self.level_attention(lvl_feat)      # (B,S,N,lvl)
            fused_weighted = (fused_stack * attn.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=3,keepdim=True)
        else:
           fused_weighted = fused_stack.mean(dim=3,keepdim=True)


        # ---- (5) 多层可见度汇总 ----
        # 可以简单平均，也可取最大层的结果
        fused_weighted = fused_weighted.reshape(B*S*N,1,g,g,g)

        # confidence_volume = self.fusion_conv(fused_weighted)
        
        confidence_volume = fused_weighted.reshape(B,S,N,g,g,g)

        visibility_pred = torch.stack(visibility_levels, dim=2).mean(dim=2)  # (B,S,N,V)

        return confidence_volume, visibility_pred
    
class AttentionTransformerWithFFN_Learnable(nn.Module):
    def __init__(self, num_heads, hidden_dim, ffn_dim, num_layers, dropout, grid_dim=8):
        super(AttentionTransformerWithFFN_Learnable, self).__init__()
        
        self.grid_dim = grid_dim
        
        # 多头自注意力层（保持不变）
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # 前馈神经网络（保持不变）
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim), 
                nn.ReLU(), 
                nn.Linear(ffn_dim, hidden_dim) 
            )
            for _ in range(num_layers)
        ])
        
        # 层归一化层（保持不变）
        self.layer_norms_attn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_norms_ffn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # === 修改：可学习位置编码 ===
        self.pe_dim_per_axis = 4  # 每个轴4维（8个离散值足够）
        self.pe_total_dim = 3 * self.pe_dim_per_axis  # 总共12维
        
        # 为每个坐标轴创建嵌入层
        self.pos_embed_x = nn.Embedding(grid_dim, self.pe_dim_per_axis)
        self.pos_embed_y = nn.Embedding(grid_dim, self.pe_dim_per_axis)  
        self.pos_embed_z = nn.Embedding(grid_dim, self.pe_dim_per_axis)

        self.pe_projection = nn.Linear(self.pe_total_dim, hidden_dim)
        
        # 输入映射层：输入维度 = 1 (置信度) + 12 (位置编码)
        self.input_projection = nn.Linear(1 , hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 1)

        """创建可学习的3D位置编码"""
        # 创建坐标网格 (0到g-1)
        coords = torch.stack(torch.meshgrid(
            torch.arange(grid_dim), 
            torch.arange(grid_dim), 
            torch.arange(grid_dim),
            indexing='ij'
        ), dim=-1).reshape(1, -1, 3).long()
        self.register_buffer('grid_indices', coords)

    def create_3d_learnable_pe(self, batch_size):
        # 分别查找三个轴的嵌入
        pe_x = self.pos_embed_x(self.grid_indices[..., 0])  # (G, pe_dim)
        pe_y = self.pos_embed_y(self.grid_indices[..., 1])
        pe_z = self.pos_embed_z(self.grid_indices[..., 2])
        
        # 合并三个轴的嵌入
        pe = torch.cat([pe_x, pe_y, pe_z], dim=-1)  # (G, 3*pe_dim)
        pe = pe.expand(batch_size, -1, -1)  # (batch_size, G, 12)
        
        return pe
        
    def forward(self, x):
        B, S, N, g, _, _, _ = x.shape
        assert g == self.grid_dim, f"输入g={g}与初始化grid_dim={self.grid_dim}不一致"
        
        # Flatten x: (B*S*N, g*g*g, 1)
        x_flat = x.reshape(B * S * N, g * g * g, 1) 

        # === 修改：使用可学习位置编码 ===
        pe = self.create_3d_learnable_pe(B * S * N)  # (BSN, G, 12)
        pe = self.pe_projection(pe)
        
        # 映射到 hidden_dim
        x = self.input_projection(x_flat)  # (BSN, G, hidden_dim)

        x = x + pe
        
        # Transformer 编码（保持不变）
        for i in range(len(self.attention_layers)):
            attn_output, _ = self.attention_layers[i](x, x, x)
            x = self.layer_norms_attn[i](attn_output + x)
            ffn_output = self.ffn_layers[i](x)
            x = self.layer_norms_ffn[i](ffn_output + x)
        
        # 最终投影和重塑
        x = self.output_projection(x)
        return x.reshape(B, S, N, g, g, g, 1)
    
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=32):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)  # shape(max_len, embed_dim)

    def forward(self, x):
        # x: (batch, S, embed_dim)
        S = x.size(1)
        return x + self.pe[:S, :]

class TemporalTransformer(nn.Module):
    def __init__(self, hidden_dim=32, num_heads=4):
        super().__init__()

        # 1D 投影
        self.input_proj = nn.Linear(1, hidden_dim)

        # 时间位置编码
        self.pe = TemporalPositionalEncoding(hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 投回 channel=1
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B*, S, 1)
        x = self.input_proj(x)   # (B*,S,hidden_dim)
        x = self.pe(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        return x
    
class VoxelFusionNet(nn.Module):
    def __init__(self, input_channels=2, hidden_dim=32, grid_dim=7, score_fusion_policy='legacy'):
        super().__init__()
        validate_score_fusion(score_fusion_policy)
        self.score_fusion_policy = score_fusion_policy
        if score_fusion_policy != 'legacy' and input_channels != 2:
            raise ValueError('Mean residual fusion requires two input branches')
        
        # ----------------------------------------------------------------
        # Layer 1
        # ----------------------------------------------------------------
        self.conv1 = nn.Conv3d(input_channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        
        # [修改点]: 使用 GroupNorm
        # num_groups: 分成几组。通常取 4 或 8。必须能整除 hidden_dim。
        # num_channels: 输入通道数 (即 hidden_dim)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=hidden_dim)
        
        self.act1 = nn.ReLU(inplace=True)
        
        # ----------------------------------------------------------------
        # Layer 2
        # ----------------------------------------------------------------
        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        
        # [修改点]: 使用 GroupNorm
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=hidden_dim)
        
        self.act2 = nn.ReLU(inplace=True)
        
        # ----------------------------------------------------------------
        # Layer 3
        # ----------------------------------------------------------------
        self.conv3 = nn.Conv3d(hidden_dim, 1, kernel_size=3, padding=1, bias=True)
        
        # 初始化
        nn.init.constant_(self.conv3.weight, 0.001 if score_fusion_policy == 'legacy' else 0.)
        nn.init.constant_(self.conv3.bias, 0)

        if score_fusion_policy == 'source_evidence_residual':
            self.source_adapter = nn.Conv3d(4, hidden_dim, kernel_size=1, bias=False)
            nn.init.zeros_(self.source_adapter.weight)

    def forward(self, x, source_evidence=None):
        # 注意：GroupNorm 不需要像 BN 那样区分 train/eval 模式
        original_scores = x
        x = self.conv1(x)
        if self.score_fusion_policy == 'source_evidence_residual':
            if source_evidence is None or source_evidence.shape != (x.shape[0], 4, *x.shape[-3:]):
                raise ValueError('Source evidence requires four aligned auxiliary volumes')
            x = x + self.source_adapter(source_evidence)
        elif source_evidence is not None:
            raise ValueError('Auxiliary source evidence requires its explicit policy')
        x = self.act1(self.norm1(x))
        x = self.act2(self.norm2(self.conv2(x)))
        out = self.conv3(x)
        return fuse_scores(original_scores, out, self.score_fusion_policy)

def bilinear_sampler(volume, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    V = volume.shape[-1]
    xgrid, ygrid, zgrid = coords.split([1,1,1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2*xgrid/(V-1) - 1
    ygrid = 2*ygrid/(V-1) - 1
    zgrid = 2*zgrid/(V-1) - 1

    grid = torch.cat([xgrid, ygrid, zgrid], dim=-1)
    volume = F.grid_sample(volume, grid, align_corners=True)  

    # if mask:
    #     mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
    #     return volume, mask.float()
    return volume

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def fuse_visibility(vis, mode='softmax', tau=0.5):
    """
    vis: (B,S,N,V)
    mode: ['softmax', 'logsumexp', 'prob_or', 'soft_or']
    return: vis3d (B,S,N)
    """
    if mode == 'softmax':
        w = torch.softmax(vis / tau, dim=-1)
        vis3d = (w * vis).sum(dim=-1)
    elif mode == 'logsumexp':
        m = vis.max(dim=-1, keepdim=True)[0]
        vis3d = tau * torch.log(torch.exp((vis - m)/tau).sum(dim=-1)) + m.squeeze(-1)
    elif mode == 'prob_or':
        vis3d = 1 - torch.prod(1 - vis.clamp(0,1), dim=-1)
    elif mode == 'soft_or':
        vis3d = torch.sigmoid(torch.logsumexp(vis * tau, dim=-1) / tau)
    else:
        raise ValueError(f"Unknown mode {mode}")
    return vis3d.unsqueeze(-1)

class Pips(nn.Module):
    def __init__(self, stride=8, grid_dim=7, load_dino=True, semantic_encoder="dinov2", semantic_cache_path="", correlation_geometry="legacy", dino_query_policy="legacy", view_weight_policy="legacy", coordinate_update_policy="legacy", score_fusion_policy="legacy"):
        super(Pips, self).__init__()

        validate_correlation_geometry(correlation_geometry)
        self.correlation_geometry = correlation_geometry
        validate_query_policy(dino_query_policy)
        self.dino_query_policy = dino_query_policy
        validate_view_weight_policy(view_weight_policy)
        self.view_weight_policy = view_weight_policy
        validate_coordinate_update(coordinate_update_policy)
        self.coordinate_update_policy = coordinate_update_policy
        validate_score_fusion(score_fusion_policy)
        self.score_fusion_policy = score_fusion_policy
        if coordinate_update_policy!='legacy' and (grid_dim<3 or grid_dim%2!=1):
            raise ValueError('Center update requires an odd grid >=3')

        self.stride = stride

        self.latent_2d = 128
        # self.latent_dim = 128
        # self.corr_levels = 3
        self.grid_dim = grid_dim
        # self.expand_factor = 2
        # self.shrink_factor = 0.5
        # self.D_stat = 8
        # self.D_scale_embed = 4

        # self.stats_encoder = GeometricStatsEncoder(output_dim=self.D_stat)
        # self.scale_embedding = nn.Embedding(self.corr_levels, self.D_scale_embed)
        
        self.fnet = SpaEncoder(output_dim=self.latent_2d, norm_fn='instance', dropout=0, stride=stride)
        # self.view_attn_mlp = ViewAttentionMLP(D_scale_embed=self.D_scale_embed)
        # self.predictor = MultiScaleCorrelationFusionWithVisibility_ScaleFirst_Attention(lvl=self.corr_levels,scale_embeddinmg=self.scale_embedding,D_stat=self.D_stat,D_scale_embed=self.D_scale_embed)
        # self.SpatialAttention = AttentionTransformerWithFFN_Learnable(num_heads=2,hidden_dim=32,ffn_dim=64,num_layers=4,dropout=0.1,grid_dim=self.grid_dim)
        # self.TemporalAttention = TemporalTransformer(hidden_dim=32,num_heads=2)

        


        if semantic_encoder not in FAMILIES:
            raise ValueError('Unsupported semantic encoder: '+str(semantic_encoder))
        if not load_dino and semantic_cache_path:
            raise ValueError('Cannot use a semantic cache when DINO is disabled')
        if load_dino and semantic_encoder == 'dinov3' and not semantic_cache_path:
            raise ValueError('DINOv3 requires a validated frozen feature cache; run export_semantic_features first')
        self.semantic_encoder = semantic_encoder
        self.semantic_cache = FrozenFeatureCache(semantic_cache_path, semantic_encoder) if semantic_cache_path else None
        self.dino_version = FAMILIES[semantic_encoder]['model_id']
        self.processor = None
        self.dino = None
        self.dinoconfig = None
        self.dino_dim = 0
        self.dino_proj = None
        if load_dino and semantic_encoder == 'dinov2':
            self.processor = AutoImageProcessor.from_pretrained(self.dino_version)
            self.dino = AutoModel.from_pretrained(self.dino_version)
            self.dinoconfig = AutoConfig.from_pretrained(self.dino_version)
            self.dino_dim = self.dinoconfig.hidden_size
            self.dino_proj = nn.Conv2d(self.dino_dim, self.latent_2d, kernel_size=1)
            self.dino.requires_grad_(False)
        elif load_dino:
            self.dino_dim = self.semantic_cache.patches.shape[3]
            self.dino_proj = nn.Conv2d(self.dino_dim, self.latent_2d, kernel_size=1)
        # self.fusion_conv = nn.Conv2d(2 * self.latent_2d, self.latent_2d, kernel_size=1)

        self.fusion_net = VoxelFusionNet(input_channels=2, hidden_dim=32, score_fusion_policy=score_fusion_policy)
        
        # self.delta_block = DeltaBlock(latent_dim=self.latent_2d, posenc_dim=self.flow_embed_dim, grid_dim=self.grid_dim, corr_levels=self.corr_levels)
        # self.ffeats_norm = nn.GroupNorm(1, self.latent_dim)
        # self.ffeats_updater = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.GELU())
        

    def forward(
        self,
        trajs_e0,
        vis_e0,
        rgbs,
        k_matrix,
        rt_matrix,
        initial_size,
        threshold,
        trajs_2d,
        vis_2d,
        iters=8,
        query_t=None,
        trajs_g=None,
        vis_g=None,
        valids=None,
        sw=None,
        feat_init=None,
        is_train=False,
        beautify=False,
        global_step=0,
        target_scale=0.25,
        lambda_smooth=10.0,
        lambda_rigid=0.2,
        adaptive_size=True,
        loss_on_final_only=False,
        use_dino=True,
        use_cotracker=True,
        loss_coordinate_policy="constrained",
        sampling_policy="native",
        reprojection_weights=None,
        search_size_policy="shared",
        fixed_support_reference=None,
        reprojection_targets_override=None,
        reprojection_visibility_override=None,
    ):
        
        '''
        0. Preparation
        '''
        validate_sampling_policy(sampling_policy)
        validate_search_size(search_size_policy)
        if loss_coordinate_policy not in ("constrained", "legacy_proposal"):
            raise ValueError(f"Unknown loss_coordinate_policy: {loss_coordinate_policy!r}")
        B,S,N,D = trajs_e0.shape
        assert(D==3)

        B,V,S,C,H,W = rgbs.shape
        rgbs_orig = rgbs.permute(0,2,1,3,4,5) #(bvschw to bsvchw)

        H8 = H//self.stride
        W8 = W//self.stride

        n_views = k_matrix.shape[2]
        assert(n_views==V)

        device = rgbs_orig.device
        g = self.grid_dim

        if self.score_fusion_policy != 'legacy' and not (use_dino and use_cotracker):
            raise ValueError('Mean residual fusion requires DINO and fnet')
        feat_dino = None
        h_feat, w_feat = None, None
        dino_scale_w, dino_scale_h = None, None
        if use_dino:
            if self.semantic_cache is None:
                patches, geometry = extract_patch_tokens(self.dino, rgbs_orig)
            else:
                patches, geometry = self.semantic_cache.get(rgbs_orig)
            feat_dino = F.normalize(patches, p=2, dim=3)
            del patches
            h_feat, w_feat = geometry['feature_size']
            dino_scale_w, dino_scale_h = geometry['image_to_feature_scale']

        # 0-2. Cotracker feat Preparation
        fmaps = None
        if use_cotracker:
            rgbs_co = 2 * (rgbs_orig / 255.0) - 1.0
            rgbs_co = rgbs_co.reshape(B*S*V,C,H,W)
            # [FP32]
            fmaps = self.fnet(rgbs_co).reshape(B,S,V,self.latent_2d, H8, W8)

        # 0-3. Camera Preparation
        # 原始 K 矩阵 (针对原图 H, W)
        k_matrix_reshape = k_matrix.reshape(B*S*V,3,3)
        
        # CoTracker 原始内参 (针对 H/8) - 仅用于提取 Query 时投影
        k_matrix_cotracker = None
        if use_cotracker:
            k_matrix_cotracker = k_matrix_reshape.clone().to(device)
            k_matrix_cotracker = camera.update_after_resize(k_matrix_cotracker,1/self.stride,1/self.stride)
            k_matrix_cotracker = k_matrix_cotracker.reshape(B,S,V,3,3)

        # DINO 原始内参 (针对 H/14) - 仅用于提取 Query 时投影
        k_matrix_dino = None
        if use_dino:
            k_matrix_dino = k_matrix_reshape.clone().to(device)
            k_matrix_dino = camera.update_after_resize(k_matrix_dino,dino_scale_w,dino_scale_h)
            if self.correlation_geometry == 'dino_patch_center':
                k_matrix_dino = patch_center_intrinsics(k_matrix_dino, (h_feat, w_feat), (H, W))
            k_matrix_dino = k_matrix_dino.reshape(B,S,V,3,3)

        coord_predictions1 = [] 
        loss_coord_predictions = []
        Origin = trajs_e0[:,0].detach()

        # 0-4.Size Preparation
        if torch.is_tensor(initial_size):
            size = initial_size.to(device=device, dtype=torch.float32)
            if size.ndim == 2:
                size = size.unsqueeze(0)
            if size.shape[-1] == 1:
                size = size.repeat(1, 1, 3)
            if size.shape != (B, N, 3):
                raise ValueError(f"initial_size tensor shape {size.shape} does not match {(B, N, 3)}")
        else:
            size = torch.tensor([initial_size,initial_size,initial_size],dtype=torch.float32).reshape(1,1,3).repeat(B,N,1).to(device)
            
        size = initialize_search_size(size, S, search_size_policy)

        '''
        1. Pre-compute Unified Query & Global Correlation Pyramids
        '''
        
        # =========================================================
        # 1-1. 提取 CoTracker Query 并计算全局相关图
        # =========================================================
        
        # A. 提取 Query (FP32)
        # 这里的 query_coords_co 仅是临时变量，其实可以不存
        low_res_corr_co = None
        if use_cotracker:
            query_feature_co = torch.zeros(B, V, N, self.latent_2d, device=device) 
            
            for s in range(S):
                mask_s = (query_t == s) 
                if not mask_s.any(): continue
                
                query_xyz_s = trajs_e0[:, 0].unsqueeze(1)
                k_s_co = k_matrix_cotracker[:, s].unsqueeze(1) 
                rt_s = rt_matrix[:, s].unsqueeze(1) 

                with torch.amp.autocast('cuda', enabled=False):
                    coords_s_co, _, _, _ = camera.batch_camera_projection(
                        query_xyz_s.float(), k_s_co.float(), rt_s.float(), h=H8, w=W8
                    )
                coords_s_co = coords_s_co.squeeze(1)
                
                feat_s = fmaps[:, s] 
                # 采样
                sampled = feature_sample_2d(coords=coords_s_co, fmaps=feat_s, sampling_policy=sampling_policy)
                mask_feat = mask_s.unsqueeze(1).unsqueeze(-1).expand(B, V, N, self.latent_2d)
                query_feature_co[mask_feat] = sampled[mask_feat]

            query_feature_co = F.normalize(query_feature_co, p=2, dim=-1)
            fmaps = F.normalize(fmaps, p=2, dim=3)

            # B. 计算 CoTracker 全局相关图 (B, S, V, N, H8, W8)
            # 优化：一次性并行计算所有 S 帧，利用 einsum 广播
            # 'bsvchw, bvnc -> bsvnhw' (同视角匹配: View v matches View v)
            # 虽然不转 half，但计算时开启 autocast 是安全的加速手段
            with torch.cuda.amp.autocast(enabled=True): 
                 low_res_corr_co = torch.einsum('bsvchw, bvnc -> bsvnhw', fmaps, query_feature_co).float()
            low_res_corr_co = torch.nan_to_num(low_res_corr_co, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

        # =========================================================
        # 1-2. 提取 DINO Query 并计算全局相关图
        # =========================================================
        
        # A. 提取 Query
        low_res_corr_dino = None
        source_bank = None
        use_source_evidence = self.score_fusion_policy == 'source_evidence_residual'
        if use_source_evidence and self.dino_query_policy != 'legacy':
            raise ValueError('Source evidence preserves the legacy averaged DINO template')
        if use_dino:
            dino_dim = feat_dino.shape[3]
            query_feature_dino = torch.zeros(B, V, N, dino_dim, device=device)
            query_projection_valid = torch.zeros(B, V, N, dtype=torch.bool, device=device) if self.dino_query_policy == 'per_view' or use_source_evidence else None
            
            for s in range(S):
                mask_s = (query_t == s)
                if not mask_s.any(): continue
                
                query_xyz_s = trajs_e0[:, 0].unsqueeze(1)
                k_s_dino = k_matrix_dino[:, s].unsqueeze(1)
                rt_s = rt_matrix[:, s].unsqueeze(1)
                
                with torch.amp.autocast('cuda', enabled=False):
                    coords_s_dino, _, _, query_geom_mask = camera.batch_camera_projection(
                        query_xyz_s.float(), k_s_dino.float(), rt_s.float(), h=h_feat, w=w_feat
                    )
                coords_s_dino = coords_s_dino.squeeze(1)
                feat_s_dino = feat_dino[:, s]
                
                sampled_dino = feature_sample_2d(coords=coords_s_dino, fmaps=feat_s_dino, sampling_policy=sampling_policy)
                mask_feat_dino = mask_s.unsqueeze(1).unsqueeze(-1).expand(B, V, N, dino_dim)
                query_feature_dino[mask_feat_dino] = sampled_dino[mask_feat_dino]
                if query_projection_valid is not None:
                    query_projection_valid = torch.where(mask_s[:,None,:], query_geom_mask[:,0,:,:,0].bool(), query_projection_valid)
            
            # B. 统一 Query (Unified Query)
            # 加权平均所有可见视角的特征
            mask_dino_weight = vis_e0.permute(0, 3, 2, 1) # (B, Vq, N, 1)
            masked_dino_feat = query_feature_dino * mask_dino_weight
            sum_dino_feat = masked_dino_feat.sum(dim=1) 
            valid_dino_weight = mask_dino_weight.sum(dim=1) + 1e-6
            unified_query_dino = sum_dino_feat / valid_dino_weight # (B, N, C)
            unified_query_dino = F.normalize(unified_query_dino, p=2, dim=-1)

            # C. 计算 DINO 全局相关图 (B, S, V, N, h, w)
            # 优化：一次性并行计算所有 S 帧
            # 'bsvchw, bnc -> bsvnhw' (跨视角匹配: Unified Query matches all Views)
            # einsum 会自动处理 query (bnc) 到 (bsvnc) 的广播
            with torch.cuda.amp.autocast(enabled=True):
                if self.dino_query_policy == 'legacy':
                    low_res_corr_dino = torch.einsum('bsvchw, bnc -> bsvnhw', feat_dino, unified_query_dino).float()
                else:
                    view_templates, _, _ = per_view_templates(query_feature_dino, vis_e0[:,0].permute(0,2,1), query_projection_valid)
                    low_res_corr_dino = torch.einsum('bsvchw, bvnc -> bsvnhw', feat_dino, view_templates).float()
            if use_source_evidence:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                    templates, source_usable = source_templates(query_feature_dino, vis_e0[:,0].permute(0,2,1), query_projection_valid)
                    source_bank = torch.stack([torch.einsum('bsvchw,bnc->bsvnhw', feat_dino, templates[:,q])
                                               for q in range(V)]).detach()

        # =========================================================
        # 1-3. 统一上采样 & 释放特征显存 (Super-Resolution & Memory Release)
        # =========================================================
        
        # 【关键步骤】立即释放重型特征图！
        # 此时我们已经拿到了相关图，原始的 384 维特征已经没有用了
        del feat_dino, fmaps
        torch.cuda.empty_cache()

        # 设定统一的目标分辨率 (Target Resolution): H/4
        # 这个尺度既比 H/14 (DINO) 细致，又比 H (原图) 省显存
        new_h, new_w = int(H * target_scale), int(W * target_scale)

        # 上采样 DINO 相关图 (B, S, V, N, H/4, W/4)
        if use_dino:
            high_res_corr_dino = F.interpolate(
                low_res_corr_dino.reshape(-1, 1, h_feat, w_feat), 
                size=(new_h, new_w), mode='bilinear', align_corners=True
            ).view(B, S, V, N, new_h, new_w)
            del low_res_corr_dino
        else:
            high_res_corr_dino = torch.zeros(B, S, V, N, new_h, new_w, device=device, dtype=rgbs_orig.dtype)

        # 上采样 CoTracker 相关图 (B, S, V, N, H/4, W/4)
        if use_cotracker:
            high_res_corr_co = F.interpolate(
                low_res_corr_co.reshape(-1, 1, H8, W8), 
                size=(new_h, new_w), mode='bilinear', align_corners=True
            ).view(B, S, V, N, new_h, new_w)
            high_res_corr_co = torch.nan_to_num(high_res_corr_co, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            del low_res_corr_co
        else:
            high_res_corr_co = torch.zeros(B, S, V, N, new_h, new_w, device=device, dtype=rgbs_orig.dtype)

        # 准备共享的高分辨率相机内参 (Shared K)
        # 这个 K 是专门为 new_h, new_w 服务的
        k_matrix_shared = k_matrix.clone() # (B, S, V, 3, 3)
        k_matrix_shared[..., 0, :] *= target_scale # fx, cx
        k_matrix_shared[..., 1, :] *= target_scale # fy, cy
        k_matrix_shared = k_matrix_shared.reshape(B, S, V, 3, 3)

        
        '''
        2. TTO Loop (Optimized: Score Sampling with Shared Projection)
        '''

        track = trajs_e0.clone() 

        for itr in range(iters):
            coords = World2Grid(track,size,g)
            
            # =========================================================
            # Step 1: 构建 3D 搜索空间
            # =========================================================
            if torch.isnan(track).any(): print('nan in track'); quit()
            
            track_32 = track.float()
            size_32 = size.float() 
            world_volume = create_boundingbox(track_32, size_32, g) 
            xyz_coords = world_volume.reshape(B, S, N*g*g*g, 3)
            
            # =========================================================
            # Step 2: 共享投影 (Shared Projection)
            # =========================================================
            # 只需一次投影，即可用于两张相关图的采样！
            # 计算量直接减半，显存也省了一份
            
            with torch.amp.autocast('cuda', enabled=False):
                volume_coords_shared, _, _, _ = camera.batch_camera_projection(
                    world_points=xyz_coords, 
                    intrinsics=k_matrix_shared.float(), # <--- 使用统一的 H/4 内参
                    extrinsics=rt_matrix.float(), 
                    h=new_h, w=new_w  # <--- 使用统一的 H/4 尺寸
                )
            
            # grid_shared: (B, S, V, N, G^3, 2)
            grid_shared = volume_coords_shared.reshape(B, S, V, N, g*g*g, 2)
            
            # =========================================================
            # Step 3: 分数采样 (Score Sampling)
            # =========================================================
            
            # 3-A. 采样 DINO 分数
            # agg_corr_dino: (B, S, N, V, G^3)
            grid_dino = grid_shared
            if use_dino and self.correlation_geometry == 'query_aligned':
                grid_dino = query_aligned_grid(grid_shared, (h_feat, w_feat), (new_h, new_w), (H, W), target_scale)
            elif use_dino and self.correlation_geometry == 'dino_patch_center':
                grid_dino = patch_center_grid(grid_shared, (h_feat, w_feat), (new_h, new_w), (H, W), target_scale)
            agg_corr_dino = sample_scores_from_heatmap(high_res_corr_dino, grid_dino, sampling_policy=sampling_policy)
            agg_corr_dino = torch.nan_to_num(agg_corr_dino, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            
            # 3-B. 采样 CoTracker 分数
            grid_co = grid_shared
            if use_cotracker and self.correlation_geometry == 'query_aligned':
                grid_co = query_aligned_grid(grid_shared, (H8, W8), (new_h, new_w), (H, W), target_scale)
            agg_corr_co = sample_scores_from_heatmap(high_res_corr_co, grid_co, sampling_policy=sampling_policy)
            agg_corr_co = torch.nan_to_num(agg_corr_co, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            
            # CoTracker 需要额外的 Mask 处理 (剔除不可见视角的匹配)
            mask_co_query = vis_e0.unsqueeze(-1) 
            agg_corr_co = agg_corr_co * mask_co_query
            
            # 注意：agg_corr_dino 和 agg_corr_co 现在已经是最终的 Correlation Score
            # 不需要再做任何点积操作

            # =========================================================
            # Step 4. Volume Aggregation (Visibility-Aware Fusion)
            # =========================================================
            
            # =========================================================
            # 4.1 可见性预测 (Visibility Prediction) - [Query-Based Switch]
            # =========================================================
            
            # 1. 提取 DINO 峰值分数
            # 经上采样后，Visible~0.6-0.8, Background~0.4-0.5
            if use_dino:
                view_conf_raw, _ = agg_corr_dino.max(dim=-1) # (B, S, N, V)
            else:
                view_conf_raw, _ = agg_corr_co.max(dim=-1)

            # 2. 定义两套 DINO 概率标准
            
            # 标准 A: 基础概率 (Base) - 只要有点像就行
            # 适用于 Query 可见的情况，因为我们确信 Query 特征是干净的
            prob_dino_base = torch.sigmoid((view_conf_raw - 0.80) * 20.0) 
            
            # 标准 B: 严格概率 (Strict) - 必须非常像
            # 适用于 Query 不可见的情况，防止背景误报 (FPR)
            prob_dino_strict = torch.sigmoid((view_conf_raw - 0.85) * 20.0)

            # 3. 准备辅助数据
            if vis_2d is not None:
                if vis_2d.shape[-1] != view_conf_raw.shape[-1]:
                    vis_co_aligned = vis_2d.permute(0, 1, 3, 2)
                else:
                    vis_co_aligned = vis_2d
            else:
                # 如果没有 vis_2d，就全 0，退化为纯 DINO
                vis_co_aligned = torch.zeros_like(view_conf_raw)

            # 4. [核心修改] 构造 Query 帧的可见性 Mask
            # vis_e0: (B, 1, N, V) -> 扩展到 (B, S, N, V)
            # 这里的逻辑是：如果该点在 Query 帧可见，那么这一整条轨迹都属于 "Easy Mode"
            is_query_visible = vis_e0.expand_as(view_conf_raw) > 0.5

            # 5. 分支决策
            
            # --- 分支 1: Query 可见 (Easy / High Quality Reference) ---
            # 策略：混合 DINO 和 CoTracker。
            # 既然 Query 可见，CoTracker 的预测通常是靠谱的，可以用来提升平滑度。
            # DINO 使用 Base 阈值，保证召回率 (Recall)。
            if use_dino and use_cotracker:
                score_easy = 0.4 * prob_dino_base + 0.6 * vis_co_aligned
            elif use_dino:
                score_easy = prob_dino_base
            elif use_cotracker:
                score_easy = vis_co_aligned
            else:
                score_easy = torch.ones_like(prob_dino_base)
            
            # --- 分支 2: Query 不可见 (Hard / Cross-View Localization) ---
            # 策略：纯 DINO 严格模式。
            # 这种情况下 CoTracker 的参考特征是坏的，vis_co_aligned 通常也是错的或全0，不能用。
            # 只能依靠 DINO 的语义跨视角能力，且必须过 Strict 阈值，否则就是误报。
            score_hard = prob_dino_strict if use_dino else score_easy
            
            # 6. 最终融合
            vis_prob_dino = is_query_visible.float() * score_easy + (~is_query_visible).float() * score_hard
            # 4.2 DINO 融合 (Softmax Weighted Aggregation)
            tau = 0.1 
            weights_dino = F.softmax(view_conf_raw / tau, dim=-1) 
            weights_dino_exp = weights_dino.unsqueeze(-1)
            if use_dino:
                final_vol_dino = (agg_corr_dino * weights_dino_exp).sum(dim=3) # (B, S, N, G^3)
            else:
                final_vol_dino = torch.zeros_like(agg_corr_dino[:, :, :, 0])

            # 4.3 fnet aggregation: optional independent view weights.
            if use_cotracker:
                raw_weights_co = fnet_view_weights(agg_corr_co, weights_dino, self.view_weight_policy, use_dino, tau)
                mask_co_valid = vis_e0 
                masked_weights_co = raw_weights_co * mask_co_valid
                weight_sum_co = masked_weights_co.sum(dim=3, keepdim=True) + 1e-6
                weights_co_final = masked_weights_co / weight_sum_co
                
                weights_co_final_exp = weights_co_final.unsqueeze(-1)
                final_vol_co = (agg_corr_co * weights_co_final_exp).sum(dim=3) # (B, S, N, G^3)
            else:
                final_vol_co = torch.zeros_like(final_vol_dino)
            
            
            '''
            5.Network Fusion & Update
            '''
            
            input_vol = torch.stack([final_vol_dino, final_vol_co], dim=1)
            input_vol = torch.nan_to_num(input_vol, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            input_vol = input_vol.permute(0, 2, 3, 1, 4).reshape(B*S*N, 2, g, g, g)
            with torch.amp.autocast('cuda', enabled=False):
                if use_source_evidence:
                    evidence = sample_source_evidence(source_bank, source_usable, grid_dino.float(), weights_dino.float(),
                                                      (new_h, new_w), sampling_policy)
                    evidence = evidence.reshape(B*S*N, 4, g, g, g)
                    # Recompute hidden fusion activations during backward; the
                    # eight-step graph otherwise exceeds 24 GB with extra evidence.
                    confidence = checkpoint.checkpoint(self.fusion_net, input_vol.float(), evidence.float(),
                                                       use_reentrant=False, preserve_rng_state=False) if torch.is_grad_enabled() else self.fusion_net(input_vol.float(), evidence.float())
                else:
                    confidence = self.fusion_net(input_vol.float())
            confidence = torch.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            
            # ... (接下来的 SoftArgmax, Delta 更新, Loss 计算等逻辑保持完全一致，无需修改) ...

            if not torch.isfinite(confidence).all():
                raise FloatingPointError("Non-finite values in fusion confidence volume.")

            # confidence_volume = confidence_volume.permute(0,2,3,4,5,1,6).reshape(B*N*g*g*g,S,1)

            
            # confidence = confidence_volume
            
            # # confidence = self.TemporalAttention(confidence_volume)

            # confidence = confidence.reshape(B,N,g,g,g,S,1).permute(0,5,1,2,3,4,6)

            # volume_coords = volume_coords.reshape(B,S,V,N,g,g,g,2)
            # visualize_voxel_correlation_heatmap(confidence_volume,volume_coords*self.stride,rgbs,itr)

             # --- find argmax in voxel space per (B,S,N) ---
            conf_flat = confidence.reshape(B*S*N, -1)  # (M, g^3)
            max_idx = conf_flat.argmax(dim=1)  # (M,)

            # convert flat idx -> z,y,x (depth, row, col) assuming ordering z*(g*g)+y*g+x
            g_sq = g * g
            max_z = (max_idx // g_sq).long()
            rem = max_idx % g_sq
            max_y = (rem // g).long()
            max_x = (rem % g).long()

            # --- padding to handle border cases: pad 1 on each side in D,H,W ---
            # padded dims = g+2 each
            pad = (1,1, 1,1, 1,1)  # (W_left,W_right, H_top,H_bottom, D_front,D_back)
            # We will pad both confidence and world_coords
            confidence_padded = F.pad(confidence, pad, mode='constant', value=-1e9)  # (M,1,g+2,g+2,g+2)

            # prepare world_coords flattened similarly
            # world_coords: (B,S,N,g,g,g,2)
            wc = world_volume.reshape(B*S*N, g, g, g, 3)  # (M,g,g,g,2)
            # we need to permute to (M,2,g,g,g) to pad spatial dims
            wc_perm = wc.permute(0, 4, 1, 2, 3).contiguous()  # (M,2,g,g,g)
            wc_padded = F.pad(wc_perm, pad, mode='replicate')  # (M,2,g+2,g+2,g+2)
            # bring back to (M, g+2, g+2, g+2, 2) for easy indexing
            wc_padded = wc_padded.permute(0, 2, 3, 4, 1).contiguous()  # (M, Dp, Hp, Wp, 2)
            # confidence_padded is (M,1,Dp,Hp,Wp)

            Dp, Hp, Wp = g+2, g+2, g+2
            # --- compute indices of 3x3x3 neighborhood for each sample vectorized ---
            # offsets in each axis: [-1,0,1]
            offsets = torch.tensor([-1, 0, 1], device=device, dtype=torch.long)
            oz, oy, ox = torch.meshgrid(offsets, offsets, offsets, indexing='ij')  # shape (3,3,3)
            oz = oz.reshape(-1)  # (27,)
            oy = oy.reshape(-1)
            ox = ox.reshape(-1)
            K = oz.shape[0]  # 27

            # centers in padded coords (add +1 due to padding)
            cz = (max_z + 1).long()  # (M,)
            cy = (max_y + 1).long()
            cx = (max_x + 1).long()

            # create (M, K) coords
            # expand centers to (M, K) and add offsets
            cz_rep = cz.unsqueeze(1) + oz.unsqueeze(0)  # (M,K)
            cy_rep = cy.unsqueeze(1) + oy.unsqueeze(0)
            cx_rep = cx.unsqueeze(1) + ox.unsqueeze(0)
            

            # clamp just in case (should be inside because of padding, but safe)
            cz_rep = cz_rep.clamp(0, Dp - 1)
            cy_rep = cy_rep.clamp(0, Hp - 1)
            cx_rep = cx_rep.clamp(0, Wp - 1)

            # compute linear indices into flattened spatial grid per sample: idx = z*(Hp*Wp) + y*(Wp) + x
            linear_idx = cz_rep * (Hp * Wp) + cy_rep * (Wp) + cx_rep  # (M,K)

            # flatten confidence_padded per sample spatial dims to (M, Dp*Hp*Wp)
            conf_flat_padded = confidence_padded.reshape(B*S*N, -1)  # (M, Dp*Hp*Wp)
            # gather neighborhood values
            # linear_idx long tensor
            neigh_vals = torch.gather(conf_flat_padded, dim=1, index=linear_idx)  # (M,K)

            # softmax over K (neighborhood)
            temperature = 1.0 / (1.0 + 0.5 * itr)
            probs = torch.softmax(neigh_vals / temperature, dim=1)  # (M,K)

            # gather corresponding world coords: wc_padded shape (M, Dp, Hp, Wp, 2) -> flatten spatial dims
            wc_flat = wc_padded.reshape(B*S*N, -1, 3)  # (M, Dp*Hp*Wp, 2)
            # index to (M, K, 2)
            idx_expand = linear_idx.unsqueeze(-1).expand(-1, -1, 3)  # (M,K,2)
            neigh_coords = torch.gather(wc_flat, dim=1, index=idx_expand)  # (M,K,2)


            # weighted sum to get predicted coords
            probs_exp = probs.unsqueeze(-1)  # (M,K,1)
            pred_coords_flat = (probs_exp * neigh_coords).sum(dim=1)  # (M,2)

            # reshape back to (B,S,N,2)
            world_coords = pred_coords_flat.reshape(B, S, N, 3)

            if self.coordinate_update_policy == 'center_evidence':
                world_coords, _ = center_evidence_update(
                    confidence, world_volume, track, world_coords, temperature)


            track = world_coords.clone()
            # track[:,0] = Origin

            B, S, N, _ = track.shape
        
            # 1. 构造时间掩码 (Mask)
            # 我们要找出一个 (B, S, N) 的 Bool 矩阵，其中 True 的位置代表该点是查询帧
            
            # 生成时间轴索引 [0, 1, ..., S-1] -> (1, S, 1)
            time_idx = torch.arange(S, device=track.device).reshape(1, S, 1)
            
            # 扩展 query_t -> (B, 1, N)
            q_t_expanded = query_t.unsqueeze(1)
            
            # 广播比较: (1, S, 1) == (B, 1, N) -> (B, S, N)
            is_query_frame = (time_idx == q_t_expanded) 
            
            # 2. 应用校正
            # 我们需要把 is_query_frame 扩展到 (B, S, N, 3) 以便进行坐标赋值
            mask_3d = is_query_frame.unsqueeze(-1).expand_as(track)
            
            # 扩展 Origin 以匹配 track 形状: (B, N, 3) -> (B, 1, N, 3) -> (B, S, N, 3)
            origin_expanded = Origin.unsqueeze(1).expand_as(track)
            
            # 使用 torch.where 进行选择性赋值
            # 如果是查询帧，强制使用 Origin (GT)；否则保持预测值 track
            track = torch.where(mask_3d, origin_expanded, track)

            
            coords_new = World2Grid(track,size,g)
            delta = coords_new - coords


            if torch.isnan(coords).any():
                print('nan in coords output by DeltaBlock.')

            

            # init_feature = init_feature + d_feats

            # Returned tracks and subsequent updates always fix query positions.
            # The explicit legacy policy is a loss-only attribution control:
            # reproduce the pre-V01 objective on uncorrected voxel proposals.
            coord_predictions1.append(track)
            loss_coord_predictions.append(
                world_coords if loss_coordinate_policy == "legacy_proposal" else track
            )

            if torch.isnan(world_coords).any():
                print('nan in world_coords output by DeltaBlock.')
                print('delta:',delta)
                print('size:',size)
                print('world_coords:',world_coords)
                quit()


            # 3-3 size change
            coords_flat = volume_coords_shared.reshape(B, S, V, N, g*g*g, 2)
            p_max, _ = coords_flat.max(dim=4) # (B, S, V, N, 2)
            p_min, _ = coords_flat.min(dim=4) # (B, S, V, N, 2)
            range_2d = p_max - p_min          # (B, S, V, N, 2)

            # 2. 取 X, Y 方向最大跨度
            spans = range_2d.max(dim=-1)[0]   # (B, S, V, N)

            # 3. 跨视图聚合 (取所有可见视角的平均，反映该时刻的平均观测质量)
            view_avg_spans = spans.max(dim=2)[0] # (B, S, N)

            # Preserve the shrink rule, changing only its time granularity.
            size = adapt_search_size(size, view_avg_spans, delta, threshold,
                                     itr, adaptive_size, search_size_policy)


        
        '''
        6. Loss & Backward
        '''
        
        # =========================================================
        # 6.1 主损失：带双重掩码的重投影损失 + 平滑 + 刚性
        # =========================================================
        
        # 输入形状确认:
        # coord_predictions1: List of (B, S, N, 3)
        # trajs_2d: (B, S, V, N, 2)
        # vis_2d:   (B, S, V, N)
        # vis_e0:   (B, 1, N, V)
        
        total_loss = 0.0
        loss_diagnostic_rows = []
        # rep_loss_total = 0.0 # 可选：用于记录分项 loss
        # smooth_loss_total = 0.0
        # rigid_loss_total = 0.0

        gamma = 0.8 
        
        # -----------------------------------------------------
        # 1. 构造双重掩码 (Double Mask)
        # -----------------------------------------------------
        
        # [调整 vis_e0]: (B, 1, N, V) -> (B, 1, V, N)
        vis_e0_aligned = vis_e0.permute(0, 1, 3, 2)
        
        # [语义掩码]: Query Valid AND Tracking Valid
        # (B, 1, V, N) * (B, S, V, N) -> (B, S, V, N)
        # Oracle diagnostics affect only the primary loss. Preserve pseudo inputs
        # used by forward fusion, rigidity and exported visibility.
        rep_targets = trajs_2d if reprojection_targets_override is None else reprojection_targets_override
        rep_visibility = vis_2d if reprojection_visibility_override is None else reprojection_visibility_override
        if rep_targets.shape != trajs_2d.shape or rep_visibility.shape != vis_2d.shape:
            raise ValueError('Reprojection override dimensions differ')
        semantic_mask = vis_e0_aligned * rep_visibility
        frame_ids = torch.arange(S, device=trajs_2d.device).view(1, S, 1, 1)
        causal_mask = (frame_ids >= query_t.view(B, 1, 1, N)).float()
        semantic_mask = semantic_mask * causal_mask
        if reprojection_weights is not None:
            if reprojection_weights.shape != semantic_mask.shape:
                raise ValueError('Reprojection weights differ from pseudo dimensions')
            if not torch.isfinite(reprojection_weights).all() or torch.any(reprojection_weights <= 0):
                raise ValueError('Reprojection weights must be finite and positive')
            semantic_mask = semantic_mask * reprojection_weights.detach()

        
        
        
        # -----------------------------------------------------
        # 2. 迭代计算 Loss
        # -----------------------------------------------------
        
        loss_predictions = loss_coord_predictions[-1:] if loss_on_final_only else loss_coord_predictions
        fixed_depth_scale = None
        if fixed_support_reference is not None:
            from track3d.utils.fixed_support_loss import input_depth_scale, fixed_support_terms
            with torch.amp.autocast('cuda', enabled=False):
                fixed_depth_scale = input_depth_scale(fixed_support_reference.float(), rt_matrix.float(), semantic_mask)
        for i, pred_track_3d in enumerate(loss_predictions):
            iter_index = iters - 1 if loss_on_final_only else i
            
            # 初始化分项 Loss，防止未定义错误
            rep_loss = torch.tensor(0.0, device=pred_track_3d.device)
            smooth_loss = torch.tensor(0.0, device=pred_track_3d.device)
            rigid_loss = torch.tensor(0.0, device=pred_track_3d.device)

            # ==========================
            # A. 重投影损失 (Reprojection Loss)
            # ==========================
            
            with torch.amp.autocast('cuda', enabled=False):
                # proj_pixel: (B, S, V, N, 2)
                # proj_mask:  (B, S, V, N, 1)
                proj_pixel, _, _, proj_mask = camera.batch_camera_projection(
                    world_points=pred_track_3d.float(),
                    intrinsics=k_matrix.float(),
                    extrinsics=rt_matrix.float(),
                    h=H, w=W
                )
            
            # diff: (B, S, V, N, 2)
            diff = proj_pixel - rep_targets
            
            # L2 距离: (B, S, V, N)
            dist = torch.norm(diff, p=1, dim=-1)
            
            # 几何掩码: (B, S, V, N)
            geom_mask = proj_mask.squeeze(-1)
            
            # Final Mask
            final_iter_mask = semantic_mask * geom_mask

            # U03: prediction geometry must not remove input supervision.
            # Leave the legacy path numerically unchanged when not enabled.
            if fixed_depth_scale is not None:
                with torch.amp.autocast('cuda', enabled=False):
                    fixed_uv, depth_penalty = fixed_support_terms(
                        pred_track_3d.float(), k_matrix.float(), rt_matrix.float(), fixed_depth_scale)
                    dist = (fixed_uv - rep_targets.float()).abs().sum(dim=-1) + depth_penalty
                final_iter_mask = semantic_mask

             # 计算全局有效点数
            global_valid_count = final_iter_mask.sum() + 1e-6
            
            # Masked Loss
            masked_dist = dist * final_iter_mask

            rep_loss = masked_dist.sum() / global_valid_count
            # Keep the real reprojection value before legacy in-place additions.
            # The optimization graph/arithmetic below intentionally stays intact.
            pure_rep_loss = rep_loss.detach().clone()

            iter_loss = rep_loss


            # ==========================
            # B. 平滑损失 (Smoothness Loss)
            # ==========================
            # S 是 coord_predictions1[0].shape[1]
            if lambda_smooth > 0 and S >= 3:
                # 1. Velocity: (B, S-1, N, 3)
                velocity = pred_track_3d[:, 1:] - pred_track_3d[:, :-1]
                
                # 2. Acceleration: (B, S-2, N, 3)
                acceleration = velocity[:, 1:] - velocity[:, :-1]
                
                # 3. Norm
                acc_norm = torch.norm(acceleration, p=1, dim=-1) # (B, S-2, N)
                # Only regularize accelerations whose three contributing frames
                # are after the query frame; otherwise backward CoTracker labels
                # can pull a track before it semantically exists.
                acc_frame_ids = torch.arange(2, S, device=pred_track_3d.device).view(1, S - 2, 1)
                smooth_mask = (acc_frame_ids >= (query_t.view(B, 1, N) + 2)).float()
                smooth_loss = (acc_norm * smooth_mask).sum() / (smooth_mask.sum() + 1e-6)

            iter_loss += lambda_smooth * smooth_loss


            # ==========================
            # C. 刚性损失 (Rigidity Loss) - [Mode Frame Anchoring]
            # ==========================
            # 只有 N > 1 且 S >= 3 且在优化后期才计算
            if lambda_rigid > 0 and N > 1 and S >= 3 and iter_index >= 2:
                # --- 1. 定义拓扑 (关键修改：基于众数帧 Mode Frame) ---
                
                # query_t: (B, N)
                # 找到出现次数最多的那个时间索引
                # torch.mode 返回 (values, indices)
                mode_values, _ = torch.mode(query_t, dim=1)
                
                # 对于 Batch=1 的 TTO，直接取第一个
                ref_t = mode_values[0].item()
                
                # [DEBUG] 打印一下看看选了哪一帧 (可选)
                # if i == 2: print(f"  [Rigid] Selected Reference Frame: {ref_t}")
                
                # 使用该“众数帧”的 3D 坐标来构建 KNN 拓扑
                # 这一帧的几何结构是最可信的 "Rest Shape"
                p_ref = pred_track_3d[:, ref_t] # (B, N, 3)
                
                # 计算距离矩阵
                dist_matrix_ref = torch.cdist(p_ref, p_ref, p=2)
                
                # 寻找最近邻
                K_neighbors = min(8, N-1)
                _, knn_indices = dist_matrix_ref.topk(K_neighbors+1, dim=2, largest=False)
                knn_indices = knn_indices[:, :, 1:] # (B, N, K) 去掉自己
                
                
                # --- 2. 计算权重 (基于 2D 轨迹的一致性) ---
                # 这部分逻辑保持不变，依然计算 neighbor_obs 和 neighbor_mask
                
                # trajs_2d: (B, S, V, N, 2) -> (B, V, S, N, 2)
                obs_bv = trajs_2d.permute(0, 2, 1, 3, 4) 
                mask_bv = vis_2d.permute(0, 2, 1, 3)     
                
                # Gather 索引扩展: (B, V, S, N, K)
                idx_exp = knn_indices.unsqueeze(1).unsqueeze(1).expand(B, V, S, N, K_neighbors)
                
                # Gather Obs
                neighbor_obs = torch.gather(
                    obs_bv.unsqueeze(4).expand(-1, -1, -1, -1, K_neighbors, 2), 
                    3, 
                    idx_exp.unsqueeze(-1).expand(-1,-1,-1,-1,-1,2)
                )
                
                # Gather Mask
                neighbor_mask = torch.gather(
                    mask_bv.unsqueeze(4).expand(-1, -1, -1, -1, K_neighbors), 
                    3, 
                    idx_exp
                )
                
                # ... (中间的 Mean/Variance 计算保持不变) ...
                # ...
                center_obs = obs_bv.unsqueeze(4) 
                diff_vec = center_obs - neighbor_obs 
                
                mask_bv_bool = mask_bv.unsqueeze(4) > 0.5
                neighbor_mask_bool = neighbor_mask > 0.5
                pair_mask = (mask_bv_bool & neighbor_mask_bool).float()
                
                valid_T = pair_mask.sum(dim=2, keepdim=True)
                is_valid_calc = (valid_T >= 3)
                
                diff_masked = diff_vec * pair_mask.unsqueeze(-1)
                vec_mean = diff_masked.sum(dim=2, keepdim=True) / (valid_T.unsqueeze(-1) + 1e-6)
                vec_dev = (diff_vec - vec_mean) * pair_mask.unsqueeze(-1)
                
                variance_sq = (vec_dev ** 2).sum(dim=-1) 
                var_per_view = variance_sq.sum(dim=2) / (valid_T.squeeze(2) + 1e-6)
                
                view_mask = is_valid_calc.squeeze(2).float()
                valid_view_count = view_mask.sum(dim=1) + 1e-6
                var_avg = (var_per_view * view_mask).sum(dim=1) / valid_view_count
                
                # Weights (假设 trajs_2d 是像素坐标，sigma 设大一点)
                sigma_rigid = 10.0
                weight_rigid = torch.exp(-var_avg / sigma_rigid)
                
                has_any_view = (view_mask.sum(dim=1) > 0).float()
                weight_rigid = weight_rigid * has_any_view
                weight_rigid = weight_rigid.detach()

                # --- 3. 计算 3D 距离方差 (Rigidity Term) ---
                
                center_3d = pred_track_3d.unsqueeze(3) # (B, S, N, 1, 3)
                
                # [关键]: 使用基于 Mode Frame 的 knn_indices 扩展到 S 维度
                idx_3d = knn_indices.unsqueeze(1).expand(-1, S, -1, -1)
                
                neighbor_3d = torch.gather(
                    pred_track_3d.unsqueeze(3).expand(-1, -1, -1, K_neighbors, -1), 
                    2, 
                    idx_3d.unsqueeze(-1).expand(-1,-1,-1,-1,3)
                )
                
                # 计算每一帧的边长
                dist_3d = torch.norm(center_3d - neighbor_3d, p=2, dim=-1)
                
                # 计算边长随时间的方差 (如果完全刚性，方差应为0)
                # Pairwise rigidity is also causal: a pair is valid only after
                # both queried tracks have appeared.
                neighbor_query_t = torch.gather(query_t, 1, knn_indices.reshape(B, -1)).reshape(B, N, K_neighbors)
                pair_start_t = torch.maximum(query_t.unsqueeze(-1), neighbor_query_t)
                rigid_frame_ids = torch.arange(S, device=pred_track_3d.device).view(1, S, 1, 1)
                rigid_time_mask = (rigid_frame_ids >= pair_start_t.unsqueeze(1)).float()
                rigid_valid_count = rigid_time_mask.sum(dim=1)
                rigid_mean = (dist_3d * rigid_time_mask).sum(dim=1) / (rigid_valid_count + 1e-6)
                dist_var_3d = (((dist_3d - rigid_mean.unsqueeze(1)) ** 2) * rigid_time_mask).sum(dim=1) / (
                    rigid_valid_count + 1e-6
                )
                dist_var_3d = torch.where(rigid_valid_count >= 2, dist_var_3d, torch.zeros_like(dist_var_3d))
                
                # --- 4. 最终刚性 Loss ---
                weighted_var = dist_var_3d * weight_rigid
                
                rigid_loss = weighted_var.sum() / (weight_rigid.sum() + 1e-6)
                iter_loss += lambda_rigid * rigid_loss


            # ==========================
            # Total Loss Aggregation
            # ==========================
            weight = gamma ** (iters - 1 - iter_index)
            total_loss += weight * iter_loss
            if getattr(self, 'collect_loss_diagnostics', False):
                values = torch.stack([pure_rep_loss, smooth_loss.detach(), rigid_loss.detach(),
                    final_iter_mask.sum().detach(), semantic_mask.sum().detach()]).float().cpu().tolist()
                rep, smooth, rigid, valid, semantic = values
                loss_diagnostic_rows.append(dict(iteration=iter_index, weight=weight,
                    rep=rep, smooth=smooth, rigid=rigid, weighted_rep=weight*rep,
                    weighted_smooth=weight*lambda_smooth*smooth,
                    weighted_rigid=weight*lambda_rigid*rigid,
                    valid_weight=valid, semantic_weight=semantic))
            
            # 用于 Debug
            # if i == iters - 1:
            #     print(f"Iter {i}: Rep={rep_loss.item():.4f}, Smooth={smooth_loss.item():.4f}, Rigid={rigid_loss.item():.4f}")
        
        '''
        最后进行一次可见性校正
        '''
        # 1. 拿到最终的 3D 轨迹
        final_traj_3d = coord_predictions1[-1] # (B, S, N, 3)
        
        # 2. 计算几何可见性 (是否在屏幕内)
        # 注意：这里需要 camera 参数，假设 k_matrix, rt_matrix 在作用域内
        with torch.no_grad():
             # 使用你自己的投影函数或 camera.batch_camera_projection
             _, _, _, geom_mask = camera.batch_camera_projection(
                    world_points=final_traj_3d.float(),
                    intrinsics=k_matrix.float(),
                    extrinsics=rt_matrix.float(),
                    h=H, w=W
             )
             # geom_mask: (B, S, V, N, 1) -> (B, S, N, V)
             geom_mask = geom_mask.squeeze(-1).permute(0, 1, 3, 2)
             
             # 3. 融合
             # vis_prob_dino: (B, S, N, V)
             # 如果几何上不可见，强制置 0
             vis_query_mask = vis_e0.permute(0, 1, 3, 2) # (B, 1, V, N)
             vis_co_processed = torch.where(vis_query_mask > 0.5, vis_2d, torch.ones_like(vis_2d))
             vis_co_final = vis_co_processed.permute(0, 1, 3, 2)
             final_causal_mask = (
                 torch.arange(S, device=final_traj_3d.device).view(1, S, 1, 1)
                 >= query_t.view(B, 1, N, 1)
             ).float()
             final_vis_output = vis_prob_dino * vis_co_final * geom_mask * final_causal_mask

        self.loss_diagnostic_rows = loss_diagnostic_rows
        return coord_predictions1, final_vis_output, total_loss, pure_rep_loss, smooth_loss, rigid_loss
