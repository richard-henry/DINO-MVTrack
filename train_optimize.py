from track3d.models.encoders import semantic_identity, load_common_initialization
from track3d.utils.fnet_updates import configure_fnet_updates, validate_fnet_updates, configure_fnet_optimizer, validate_fnet_lr
from track3d.utils.fnet_updates import validate_fnet_warmup, configure_fnet_warmup, apply_fnet_stage, fnet_stage_metadata
from track3d.utils.tto_resume import resolve_schedule, restore_tto_state, carry_selected_checkpoint
from track3d.utils.loss_convergence import load_recipe, ProtocolSchedule, LossPlateau, summarize_iterations
from track3d.utils.progress_convergence import ProgressSchedule, ProgressMonitor
from track3d.utils.search_size import validate_search_size, SEARCH_SIZE_CONTRACT
from track3d.utils.dynamic_completion import complete_tracks, validate_completion, COMPLETION_CONTRACT
from track3d.utils.pseudo_confidence import epipolar_confidence, validate_supervision, SUPERVISION_CONTRACT, supervision_contract
import time
import json
import random
from track3d.utils.deterministic_sampling import validate_sampling_policy
from track3d.utils.anchor_loss import triangulation_anchor, validate_anchor, anchor_metadata
import numpy as np
import saverloader
from nets.pips2 import Pips
from nets import camera
import utils.improc
import utils.geom
import utils.misc
import utils.basic
from datasets.exportdataset_3d import ExportDataset 
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
import cv2
import os
import shutil
from eval_utils import ExperimentLogger, evaluate_batch_comparison
from track3d.utils.reproducibility import (
    configure_numerics, numerical_settings, scheduler_horizon,
    state_digest, gradient_digest, write_trace,
)


cv2.setNumThreads(0)

# ==========================================
# [新增] 3D -> 2D 投影函数 (PyTorch)
# ==========================================
def project_points_3d_to_2d(points_3d, K, RT):
    """
    将 3D 世界坐标投影到 2D 像素坐标
    points_3d: (B, S, N, 3)
    K: (B, S, 3, 3)
    RT: (B, S, 4, 4)
    """
    B, S, N, _ = points_3d.shape
    ones = torch.ones(B, S, N, 1, device=points_3d.device, dtype=points_3d.dtype)
    points_h = torch.cat([points_3d, ones], dim=-1)
    
    # Extrinsics: (B, S, 4, 4) x (B, S, N, 4, 1) -> (B, S, N, 4, 1)
    points_cam_h = torch.matmul(RT.unsqueeze(2), points_h.unsqueeze(-1))
    points_cam = points_cam_h[..., :3, :] 
    
    # Intrinsics: (B, S, 3, 3) x (B, S, N, 3, 1) -> (B, S, N, 3, 1)
    points_pix_h = torch.matmul(K.unsqueeze(2), points_cam)
    
    points_pix_h = points_pix_h.squeeze(-1) 
    z = points_pix_h[..., 2:3]
    xy = points_pix_h[..., :2]
    
    uv = xy / (z + 1e-6)
    return uv

# ==========================================
# [重构] 可视化函数：原图背景 + 实心/空心点
# ==========================================
def visualize_multi_view_images(rgbs, trajs_g_3d, trajs_e_3d, vis_g, K, RT, query_t, save_root, step):
    """
    保存多视角、多帧的对比图片。
    背景为原视频帧。
    [新增参数] query_t: (B, N) 查询时间索引
    """
    # 1. 创建目录
    step_dir = os.path.join(save_root, f"step_{step:05d}")
    os.makedirs(step_dir, exist_ok=True)
    
    # 2. 准备数据 (取 Batch 0)
    b = 0
    B, V, S, C, H, W = rgbs.shape
    N = trajs_g_3d.shape[2]
    
    # 转换 Tensor 为 Numpy (float32)
    video_numpy = rgbs[b].permute(0, 1, 3, 4, 2).detach().float().cpu().numpy()
    
    # 遍历所有视角
    for v in range(V):
        # --- A. 准备背景图像 ---
        frames_v = video_numpy[v] 
        
        # 归一化处理
        if frames_v.max() > 2.0: 
            frames_v = np.clip(frames_v, 0, 255).astype(np.uint8)
        else:
            if frames_v.min() < 0: 
                frames_v = (frames_v + 1) / 2.0
            frames_v = np.clip(frames_v * 255.0, 0, 255).astype(np.uint8)

        # --- B. 投影轨迹 (3D -> 2D) ---
        K_v = K[b, :, v]   
        RT_v = RT[b, :, v] 
        
        # 投影
        gt_2d = project_points_3d_to_2d(trajs_g_3d[b:b+1], K_v.unsqueeze(0), RT_v.unsqueeze(0))
        gt_2d = gt_2d[0].detach().float().cpu().numpy() 
        
        pred_2d = project_points_3d_to_2d(trajs_e_3d[b:b+1], K_v.unsqueeze(0), RT_v.unsqueeze(0))
        pred_2d = pred_2d[0].detach().float().cpu().numpy() 
        
        vis_v = vis_g[b, :, :, v].detach().float().cpu().numpy()
        
        # 获取当前 Batch 所有点的 Query Frame 索引 (转为 list 或 numpy)
        # query_t: (B, N) -> (N,)
        q_times = query_t[b].detach().cpu().numpy()

        # --- C. 逐帧绘制 ---
        for s in range(S):
            # 获取背景图
            canvas = cv2.cvtColor(frames_v[s], cv2.COLOR_RGB2BGR)
            
            # 画点
            for n in range(N):
                # ==========================================
                # [关键修改] 因果掩码可视化 (Causal Masking)
                # ==========================================
                # 如果当前帧 s 早于该点的 query_frame，说明该点还没"出生"，不画
                if s < q_times[n]:
                    continue

                # 下面是正常的绘制逻辑
                
                # --- 1. 绘制 GT (绿色) ---
                if vis_v[s, n] > 0.5:
                    thickness = -1 # 实心
                    radius = 3
                else:
                    thickness = 1  # 空心
                    radius = 2
                
                u_g, v_g = int(gt_2d[s, n, 0]), int(gt_2d[s, n, 1])
                if 0 <= u_g < W and 0 <= v_g < H:
                    cv2.circle(canvas, (u_g, v_g), radius, (0, 255, 0), thickness)

                # --- 2. 绘制 Pred (红色) ---
                u_p, v_p = int(pred_2d[s, n, 0]), int(pred_2d[s, n, 1])
                if 0 <= u_p < W and 0 <= v_p < H:
                    cv2.circle(canvas, (u_p, v_p), radius, (0, 0, 255), thickness)

            # --- D. 添加文字信息 ---
            info = f"Step:{step} View:{v} Frame:{s}"
            cv2.putText(canvas, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(canvas, "GT (Green)", (10, H-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(canvas, "Pred (Red)", (10, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # --- E. 保存 ---
            save_name = f"view_{v:02d}_frame_{s:03d}.jpg"
            save_path = os.path.join(step_dir, save_name)
            cv2.imwrite(save_path, canvas)
            
    print(f"📸 Saved images to {step_dir}")
# ==========================================

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params, lr_scales=None):
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr if lr_scales is None else [lr*scale for scale in lr_scales],
        num_steps+100, pct_start=0.05, cycle_momentum=False, anneal_strategy='cos')
    return optimizer, scheduler

def grads_are_finite(parameters):
    for p in parameters:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_run_config(path, config):
    os.makedirs(path, exist_ok=True)
    config_path = os.path.join(path, "config.yaml")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    print(f"Saved run config to {config_path}")

def save_prediction_npz(save_dir, step, vis_data, final=False, visibility_source="model",
                        numerical_policy="inherit"):
    os.makedirs(save_dir, exist_ok=True)
    name = "final_tracks.npz" if final else f"step-{step:06d}_tracks.npz"
    path = os.path.join(save_dir, name)
    visibility = select_prediction_visibility(vis_data, visibility_source)
    visibility_np = _prediction_numpy(visibility)
    np.savez_compressed(
        path,
        trajs_3d=vis_data["trajs_e_3d"].detach().float().cpu().numpy(),
        visibility=visibility_np,
        pred_visibility=visibility_np,
        pred_visibilities_any_view=visibility_np,
        model_visibility=vis_data["vis_prob_dino"].detach().float().cpu().numpy(),
        init_visibility=vis_data["init_visibility"].detach().float().cpu().numpy(),
        geometry_fused_visibility=vis_data["geometry_fused_visibility"].detach().float().cpu().numpy(),
        init_confidence=vis_data["init_confidence"].detach().float().cpu().numpy(),
        init_reprojection_error_px=vis_data["init_reproj_error"].detach().float().cpu().numpy(),
        init_num_views=vis_data["init_num_views"].detach().float().cpu().numpy(),
        init_ray_angle_deg=vis_data["init_ray_angle"].detach().float().cpu().numpy(),
        init_raw_triangulated_mask=vis_data["init_raw_valid"].detach().float().cpu().numpy(),
        query_points=vis_data["query_points"].detach().cpu().numpy(),
        sample_indices=_tensor_to_numpy(vis_data.get("sample_indices")),
        visibility_source=np.array(visibility_source),
        step=np.array(step, dtype=np.int64),
        prediction_contract=np.array("post_update_eval_v1"),
        prediction_mode=np.array("eval"),
        numerical_policy=np.array(numerical_policy),
        semantic_identity_json=np.array(vis_data.get("semantic_identity_json", "{}")),
        fnet_update_policy=np.array(vis_data.get('fnet_update_policy', 'legacy')),
        fnet_lr_scale=np.array(vis_data.get('fnet_lr_scale', 1.0)),
        **({k: np.array(vis_data[k]) for k in ('fnet_warmup_steps', 'fnet_stage')}
           if 'fnet_warmup_steps' in vis_data else {}),
        numerical_settings_json=np.array(json.dumps(numerical_settings(), sort_keys=True)),
        loss_coordinate_policy=np.array(vis_data.get('loss_coordinate_policy', 'constrained')),
        sampling_policy=np.array(vis_data.get('sampling_policy', 'native')),
        anchor_policy=np.array(vis_data.get('anchor_policy', 'legacy')),
        **anchor_metadata(vis_data.get('anchor_policy', 'legacy')),
        supervision_policy=np.array(vis_data.get('supervision_policy', 'legacy')),
        supervision_contract_json=np.array(vis_data.get('supervision_contract_json', json.dumps(supervision_contract(vis_data.get('supervision_policy', 'legacy')), sort_keys=True))),
        search_size_policy=np.array(vis_data.get('search_size_policy', 'shared')),
        search_size_contract_json=np.array(json.dumps(SEARCH_SIZE_CONTRACT, sort_keys=True)),
        completion_policy=np.array(vis_data.get('completion_policy', 'legacy')),
        completion_contract_json=np.array(vis_data.get('completion_contract_json', json.dumps(COMPLETION_CONTRACT, sort_keys=True))),
        lambda_triangulation_anchor=np.array(vis_data.get('lambda_triangulation_anchor', 0.0)),
        init_raw_source=np.asarray(vis_data.get('init_raw_source', 'unavailable')),
        **({'oracle_diagnostic': np.array(True),
            'input_protocol': np.array('gt_oracle_reprojection_loss_only')}
           if vis_data.get('supervision_policy', '').startswith('oracle_') else {}),
        **{key: _tensor_to_numpy(vis_data[key]) for key in (
            'init_raw_tracks_3d', 'init_raw_confidence', 'init_raw_reproj_error',
            'init_raw_num_views', 'init_raw_ray_angle', 'init_completion_mask',
            'completion_initial_tracks_3d', 'completion_anchor_tracks_3d', 'completion_motion_prior',
            'completion_supporting_views', 'completion_fallback_points',
            'init_completion_confidence', 'reprojection_weights', 'pseudo_epipolar_error_px', 'pseudo_epipolar_peer_count',
            'directional_anchor_weight', 'directional_anchor_weakness',
            'directional_anchor_scale', 'directional_anchor_support',
            'retracking_targets', 'retracking_mask',
        ) if key in vis_data},
        **({'retracking_input_sha256': np.asarray(vis_data['retracking_input_sha256'])}
           if 'retracking_input_sha256' in vis_data else {}),
    )
    print(f"Saved prediction to {path} (visibility_source={visibility_source})")

def select_prediction_visibility(vis_data, visibility_source):
    source = str(visibility_source).lower()
    if source == "model":
        return vis_data["vis_prob_dino"]
    if source == "triangulation":
        init_visibility = vis_data.get("init_visibility")
        if init_visibility is None:
            raise ValueError("visibility_source='triangulation' requires triangulation_init_root/init_visibility")
        return init_visibility
    if source == "geometry_fused":
        geometry_fused = vis_data.get("geometry_fused_visibility")
        if geometry_fused is None:
            raise ValueError("visibility_source='geometry_fused' requires triangulation metadata")
        return geometry_fused
    raise ValueError(f"Unknown visibility_source={visibility_source!r}; expected 'model', 'triangulation', or 'geometry_fused'")

def compute_geometry_fused_visibility(
    trajs_e,
    vis_prob_model,
    init_visibility,
    init_confidence,
    init_reproj_error,
    init_num_views,
    init_ray_angle,
    init_raw_valid,
    cotracker_vis,
    k_matrix,
    rt_matrix,
    query_t,
    H,
    W,
):
    B, S, N, _ = trajs_e.shape
    with torch.no_grad():
        _, _, _, geom_mask = camera.batch_camera_projection(
            world_points=trajs_e.float(),
            intrinsics=k_matrix.float(),
            extrinsics=rt_matrix.float(),
            h=H,
            w=W,
        )
        geom_any = geom_mask.squeeze(-1).permute(0, 1, 3, 2).amax(dim=-1)
        cotracker_any = cotracker_vis.float().amax(dim=2)
        reproj_quality = torch.exp(-torch.nan_to_num(init_reproj_error.float(), nan=12.0, posinf=12.0) / 6.0)
        view_quality = (init_num_views.float().clamp(0.0, 4.0) / 4.0)
        angle_quality = (init_ray_angle.float().clamp(0.0, 5.0) / 5.0)
        triang_quality = (
            0.45 * init_confidence.float().clamp(0.0, 1.0)
            + 0.20 * reproj_quality.clamp(0.0, 1.0)
            + 0.20 * view_quality
            + 0.15 * angle_quality
        )
        fused = (
            0.45 * init_visibility.float().clamp(0.0, 1.0)
            + 0.25 * triang_quality.clamp(0.0, 1.0)
            + 0.15 * init_raw_valid.float().clamp(0.0, 1.0)
            + 0.10 * cotracker_any.clamp(0.0, 1.0)
            + 0.05 * vis_prob_model.float().amax(dim=-1).clamp(0.0, 1.0)
        )
        causal_mask = (
            torch.arange(S, device=trajs_e.device).view(1, S, 1)
            > query_t.view(B, 1, N)
        ).float()
        return (fused * geom_any.float() * causal_mask).clamp(0.0, 1.0)

def _tensor_to_numpy(value):
    if value is None:
        return np.array([], dtype=np.int64)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def _prediction_numpy(value):
    array = _tensor_to_numpy(value).astype(np.float32)
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    return array

def run_model(
    model,
    d,
    stride,
    device,
    size,
    threshold,
    iters=8,
    is_train=True,
    step=0,
    target_scale=0.25,
    lambda_smooth=10.0,
    lambda_rigid=0.2,
    adaptive_size=True,
    loss_on_final_only=False,
    use_dino=True,
    use_cotracker=True,
    lambda_triangulation_anchor=0.0,
    loss_coordinate_policy="constrained",
    sampling_policy="native",
    anchor_policy="legacy",
    supervision_policy="legacy",
    completion_policy="legacy",
    search_size_policy="shared",
):
    validate_anchor(anchor_policy, lambda_triangulation_anchor)
    validate_supervision(supervision_policy)
    validate_completion(completion_policy)
    validate_search_size(search_size_policy)
    metrics = {}

    # ... (数据加载部分不变) ...
    rgbs = d['rgbs'].float().to(device)             
    k_matrix = d['intrinsics'].float().to(device)   
    rt_matrix = d['extrinsics'].float().to(device)  
    query_points = d['query_points'].float().to(device) 
    sample_indices = d.get('sample_indices', None)
    
    trajs_g = d['track_g'].float().to(device)       
    vis_g = d['visibility'].float().to(device)      

    cotracker_tracks = d['cotracker_tracks'].float().to(device) 
    cotracker_vis = d['cotracker_vis'].float().to(device)       

    # 注意：这里 trajs_2d 变成了 (B, S, V, N, 2)，符合评测要求
    trajs_2d = cotracker_tracks.permute(0, 2, 1, 3, 4) 
    vis_2d = cotracker_vis.permute(0, 2, 1, 3)

    # ... (Query 处理部分不变) ...
    query_t = query_points[:, :, 0].long() 
    query_xyz = query_points[:, :, 1:]     

    B, V, S, C, H, W = rgbs.shape
    N = query_points.shape[1]
    
    init_tracks = d.get('init_tracks_3d', None)
    init_confidence = d.get('init_confidence', None)
    init_size = d.get('init_size', None)
    init_visibility = d.get('init_visibility', None)
    init_reproj_error = d.get('init_reproj_error', None)
    init_num_views = d.get('init_num_views', None)
    init_ray_angle = d.get('init_ray_angle', None)
    init_raw_valid = d.get('init_raw_valid', None)
    if init_tracks is not None:
        trajs_e0 = init_tracks.float().to(device)
        if trajs_e0.ndim == 5 and trajs_e0.shape[0] == 1:
            trajs_e0 = trajs_e0[0]
        if trajs_e0.shape[1] != S or trajs_e0.shape[2] != N:
            raise ValueError(f"init_tracks_3d shape {trajs_e0.shape} does not match S={S}, N={N}")
        frames = torch.arange(S, device=device).view(1, S, 1, 1)
        causal_init_mask = frames > query_t.view(B, 1, N, 1)
        trajs_e0 = torch.where(causal_init_mask, trajs_e0, query_xyz.unsqueeze(1))
        if init_confidence is not None:
            init_confidence = init_confidence.float().to(device)
            if init_confidence.ndim == 4 and init_confidence.shape[0] == 1:
                init_confidence = init_confidence[0]
            init_confidence = init_confidence[:, :S, :N] if init_confidence.ndim == 3 else init_confidence[:S, :N].unsqueeze(0)
            init_confidence = torch.where(causal_init_mask.squeeze(-1), init_confidence, torch.zeros_like(init_confidence))
        if init_visibility is not None:
            init_visibility = init_visibility.float().to(device)
            if init_visibility.ndim == 4 and init_visibility.shape[0] == 1:
                init_visibility = init_visibility[0]
            init_visibility = init_visibility[:, :S, :N] if init_visibility.ndim == 3 else init_visibility[:S, :N].unsqueeze(0)
            init_visibility = torch.where(causal_init_mask.squeeze(-1), init_visibility, torch.zeros_like(init_visibility))
        init_reproj_error = _prepare_init_scalar(init_reproj_error, device, S, N, causal_init_mask.squeeze(-1), fill_value=float("nan"))
        init_num_views = _prepare_init_scalar(init_num_views, device, S, N, causal_init_mask.squeeze(-1), fill_value=0.0)
        init_ray_angle = _prepare_init_scalar(init_ray_angle, device, S, N, causal_init_mask.squeeze(-1), fill_value=0.0)
        init_raw_valid = _prepare_init_scalar(init_raw_valid, device, S, N, causal_init_mask.squeeze(-1), fill_value=0.0)
        if init_size is not None:
            init_size = init_size.float().to(device)
            if init_size.ndim == 2:
                init_size = init_size.unsqueeze(0)
            init_size = init_size[:, :N]
    else:
        trajs_e0 = query_xyz.unsqueeze(1).repeat(1, S, 1, 1) 
        init_confidence = None
        init_size = None
        init_visibility = None
        init_reproj_error = None
        init_num_views = None
        init_ray_angle = None
        init_raw_valid = None

    idx_gather = query_t.unsqueeze(1).unsqueeze(-1).expand(B, 1, N, V)
    vis_at_query = torch.gather(vis_g, 1, idx_gather).squeeze(1) 
    vis_e0 = vis_at_query.unsqueeze(1) 

    # Keep the parent filled anchor objective even when model initialization changes.
    anchor_reference = trajs_e0
    completion_data = {}
    if completion_policy == 'ray_motion':
        if init_tracks is None or init_raw_valid is None:
            raise ValueError('ray_motion requires checked triangulation initialization')
        rows = [complete_tracks(
            initial=trajs_e0[b].detach().cpu().numpy(), raw_valid=init_raw_valid[b].detach().cpu().numpy(),
            query_points=d['query_points'][b].detach().cpu().numpy(),
            intrinsics=d['intrinsics'][b].detach().cpu().numpy(), extrinsics=d['extrinsics'][b].detach().cpu().numpy(),
            pseudo_tracks=d['cotracker_tracks'][b].detach().cpu().numpy().transpose(1,0,2,3),
            pseudo_visibility=d['cotracker_vis'][b].detach().cpu().numpy().transpose(1,0,2),
            query_visibility=vis_at_query[b].detach().cpu().numpy().T, image_size=(H,W)) for b in range(B)]
        completion_data = {key: torch.from_numpy(np.stack([r[source] for r in rows]))
            for key,source in [('completion_initial_tracks_3d','tracks'), ('completion_motion_prior','motion_prior'),
                               ('completion_supporting_views','supporting_views'), ('completion_fallback_points','fallback_points')]}
        completion_data['completion_anchor_tracks_3d'] = anchor_reference.detach().cpu()
        trajs_e0 = completion_data['completion_initial_tracks_3d'].to(device)

    supervision_extra = {}
    supervision_data = {}
    if supervision_policy == 'epipolar':
        # Deterministic CPU calculation from fixed input only; no model residual,
        # GT tracks, init completion coordinates or learned confidence involved.
        rows = [epipolar_confidence(
            d['intrinsics'][b].detach().cpu().numpy(), d['extrinsics'][b].detach().cpu().numpy(),
            d['cotracker_tracks'][b].detach().cpu().numpy().transpose(1, 0, 2, 3),
            d['cotracker_vis'][b].detach().cpu().numpy().transpose(1, 0, 2),
            d['query_points'][b, :, 0].detach().cpu().numpy()) for b in range(B)]
        supervision_data = {key: torch.from_numpy(np.stack([row[source] for row in rows]))
            for key, source in [('reprojection_weights', 'weights'), ('pseudo_epipolar_error_px', 'error_px'),
                                ('pseudo_epipolar_peer_count', 'peer_count')]}
        supervision_extra['reprojection_weights'] = supervision_data['reprojection_weights'].to(device)

    if supervision_policy == 'fixed_support':
        supervision_extra['fixed_support_reference'] = anchor_reference.detach()

    if supervision_policy.startswith('oracle_'):
        from track3d.utils.oracle_supervision import oracle_targets
        targets, mask = oracle_targets(trajs_g, vis_g, k_matrix, rt_matrix,
            vis_2d, vis_e0, query_t, supervision_policy)
        supervision_extra['reprojection_targets_override'] = targets
        supervision_extra['reprojection_visibility_override'] = mask

    # [修改] 接收 vis_prob_dino
    # 确保你的 pips2.py 的 forward 函数返回了这个变量
    preds, vis_prob_dino, total_loss, rep_loss, smooth_loss, rigid_loss = model(
        trajs_e0=trajs_e0, 
        vis_e0=vis_e0, 
        rgbs=rgbs,
        k_matrix=k_matrix,
        rt_matrix=rt_matrix,
        initial_size=init_size if init_size is not None else size,
        threshold=threshold,
        trajs_2d=trajs_2d,  
        vis_2d=vis_2d,      
        iters=iters, 
        query_t=query_t,
        trajs_g=trajs_g,    
        vis_g=vis_g,
        is_train=is_train,
        global_step=step,
        target_scale=target_scale,
        lambda_smooth=lambda_smooth,
        lambda_rigid=lambda_rigid,
        adaptive_size=adaptive_size,
        loss_on_final_only=loss_on_final_only,
        use_dino=use_dino,
        use_cotracker=use_cotracker,
        loss_coordinate_policy=loss_coordinate_policy, sampling_policy=sampling_policy,
        search_size_policy=search_size_policy,
        **supervision_extra,
    )

    trajs_e = preds[-1] 
    init_visibility_for_fusion = init_visibility if init_visibility is not None else torch.zeros(B, S, N, device=device)
    init_confidence_for_fusion = init_confidence if init_confidence is not None else torch.zeros(B, S, N, device=device)
    init_reproj_for_fusion = init_reproj_error if init_reproj_error is not None else torch.full((B, S, N), float("nan"), device=device)
    init_num_views_for_fusion = init_num_views if init_num_views is not None else torch.zeros(B, S, N, device=device)
    init_ray_angle_for_fusion = init_ray_angle if init_ray_angle is not None else torch.zeros(B, S, N, device=device)
    init_raw_valid_for_fusion = init_raw_valid if init_raw_valid is not None else torch.zeros(B, S, N, device=device)
    geometry_fused_visibility = compute_geometry_fused_visibility(
        trajs_e=trajs_e,
        vis_prob_model=vis_prob_dino,
        init_visibility=init_visibility_for_fusion,
        init_confidence=init_confidence_for_fusion,
        init_reproj_error=init_reproj_for_fusion,
        init_num_views=init_num_views_for_fusion,
        init_ray_angle=init_ray_angle_for_fusion,
        init_raw_valid=init_raw_valid_for_fusion,
        cotracker_vis=vis_2d,
        k_matrix=k_matrix,
        rt_matrix=rt_matrix,
        query_t=query_t,
        H=H,
        W=W,
    )
    anchor_loss = torch.tensor(0.0, device=device)
    anchor_data = {}
    if lambda_triangulation_anchor > 0:
        if anchor_policy == 'raw_only' and (init_tracks is None or init_confidence is None):
            raise ValueError('raw_only anchor requires triangulation initialization')
        if anchor_policy == 'weak_geometry':
            from track3d.utils.directional_anchor import geometry, loss as directional_loss
            required = ('init_raw_tracks_3d','init_raw_confidence','init_raw_reproj_error',
                        'init_raw_num_views','init_raw_ray_angle')
            if init_raw_valid is None or any(d.get(k) is None for k in required):
                raise ValueError('weak_geometry requires checked raw triangulation metadata')
            fixed = geometry(d['init_raw_tracks_3d'], init_raw_valid,
                d['init_raw_confidence'],d['init_raw_reproj_error'],d['init_raw_num_views'],
                d['init_raw_ray_angle'],query_t,k_matrix,rt_matrix,vis_2d,vis_e0,(H,W))
            anchor_loss = directional_loss(trajs_e, fixed)
            anchor_data = {'directional_anchor_'+k:fixed[k] for k in ('weight','weakness','scale','support')}
            total_loss = total_loss + lambda_triangulation_anchor * anchor_loss
        elif init_tracks is not None and init_confidence is not None:
            anchor_loss, _ = triangulation_anchor(
                trajs_e, anchor_reference, init_confidence, query_t, policy=anchor_policy,
                raw_valid=init_raw_valid, raw_tracks=d.get('init_raw_tracks_3d'),
                raw_confidence=d.get('init_raw_confidence'),
            )
            total_loss = total_loss + lambda_triangulation_anchor * anchor_loss

    auxiliary_data = {}
    auxiliary_value = None
    if supervision_policy == 'retracking_aux':
        from track3d.utils.retracking_auxiliary import CONTRACT as AUX, auxiliary_loss
        keys = ('retracking_targets', 'retracking_mask', 'retracking_input_sha256')
        if any(key not in d for key in keys):
            raise ValueError('retracking_aux requires a validated candidate sidecar')
        auxiliary_data = {key: d[key] for key in keys}
        identity = list(d['retracking_input_sha256'])
        if getattr(model, 'retracking_input_sha256', identity) != identity:
            raise ValueError('Auxiliary candidates changed during training or reload')
        model.retracking_input_sha256 = identity
        auxiliary_value, auxiliary_observation = auxiliary_loss(
            trajs_e, k_matrix, rt_matrix, d['retracking_targets'], d['retracking_mask'])
        total_loss = total_loss + AUX['weight'] * auxiliary_value

    # ... (Metrics 计算部分不变) ...
    with torch.no_grad():
        l1_dists = torch.abs(trajs_e - trajs_g).sum(dim=-1) 
        metrics['l1_avg'] = l1_dists.mean().item()
        metrics['total_loss'] = total_loss.item()
        metrics['rep_loss'] = rep_loss.item()
        metrics['smooth_loss'] = smooth_loss.item()
        metrics['rigid_loss'] = rigid_loss.item()
        metrics['anchor_loss'] = anchor_loss.item()
        if getattr(model, 'collect_loss_diagnostics', False):
            metrics['loss_diagnostics'] = summarize_iterations(model.loss_diagnostic_rows,
                anchor_loss.item(), lambda_triangulation_anchor, total_loss.item())
            if auxiliary_value is not None:
                from track3d.utils.retracking_auxiliary import add_diagnostics
                metrics['auxiliary_loss'] = auxiliary_value.item()
                metrics['loss_diagnostics'] = add_diagnostics(metrics['loss_diagnostics'],
                    auxiliary_value.item(), auxiliary_observation)
            model.last_loss_diagnostics = metrics['loss_diagnostics']

    # [修改] 打包所有评测需要的数据到 vis_data
    vis_data = {
        'rgbs': rgbs,           
        'trajs_g_3d': trajs_g,  
        'trajs_e_3d': trajs_e,  
        'vis_prob_dino': vis_prob_dino, # [新增] TTO 预测的可见性
        'vis_g': vis_g,         
        'K': k_matrix,          
        'RT': rt_matrix,
        # [新增] CoTracker 数据 (用于对比)
        'cotracker_uv': trajs_2d,   # (B, S, V, N, 2)
        'cotracker_vis': vis_2d,    # (B, S, V, N)
        'H': H, # [新增] 图像尺寸
        'W': W,
        'query_t': query_t,  # <--- [新增] 必须加这一行
        'semantic_identity_json': json.dumps(semantic_identity(model), sort_keys=True),
        'fnet_update_policy': getattr(model, 'fnet_update_policy', 'legacy'),
        'fnet_lr_scale': getattr(model, 'fnet_lr_scale', 1.0),
        **fnet_stage_metadata(model),
        'query_points': query_points,
        'loss_coordinate_policy': loss_coordinate_policy,
        'sampling_policy': sampling_policy,
        'anchor_policy': anchor_policy,
        **anchor_metadata(anchor_policy),
        'search_size_policy': search_size_policy,
        'search_size_contract_json': json.dumps(SEARCH_SIZE_CONTRACT, sort_keys=True),
        'completion_policy': completion_policy,
        'completion_contract_json': json.dumps(COMPLETION_CONTRACT, sort_keys=True),
        'supervision_policy': supervision_policy,
        'supervision_contract_json': json.dumps(supervision_contract(supervision_policy), sort_keys=True),
        'lambda_triangulation_anchor': lambda_triangulation_anchor,
        **supervision_data,
        **completion_data,
        **anchor_data,
        **auxiliary_data,
        'sample_indices': sample_indices,
        'init_tracks_3d': trajs_e0,
        'init_confidence': init_confidence_for_fusion,
        'init_visibility': init_visibility_for_fusion,
        'init_size': init_size if init_size is not None else torch.full((B, N, 3), float(size), device=device),
        'init_reproj_error': init_reproj_for_fusion,
        'init_num_views': init_num_views_for_fusion,
        'init_ray_angle': init_ray_angle_for_fusion,
        'init_raw_valid': init_raw_valid_for_fusion,
        'geometry_fused_visibility': geometry_fused_visibility,
    }
    # Checked raw observations also supply the opt-in V02A anchor support.
    for key in ('init_raw_tracks_3d', 'init_raw_confidence', 'init_raw_reproj_error',
                'init_raw_num_views', 'init_raw_ray_angle', 'init_completion_mask',
                'init_completion_confidence', 'init_raw_source'):
        if key in d:
            vis_data[key] = d[key]

    return total_loss, metrics, vis_data

def _prepare_init_scalar(value, device, S, N, causal_mask, fill_value):
    if value is None:
        return None
    value = value.float().to(device)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    value = value[:, :S, :N] if value.ndim == 3 else value[:S, :N].unsqueeze(0)
    fill = torch.full_like(value, float(fill_value))
    return torch.where(causal_mask, value, fill)

def predict_for_export(model, sample, stride, device, size, threshold, amp_dtype, **kwargs):
    """Export current weights in eval mode without consuming the training RNG."""
    modes = [(module, module.training) for module in model.modules()]
    devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad(), torch.amp.autocast(
            'cuda', dtype=amp_dtype, enabled=bool(devices)
        ):
            return run_model(model, sample, stride, device, size, threshold, is_train=False, **kwargs)[2]
    finally:
        for module, training in modes:
            module.training = training


def save_tto_checkpoint(path, optimizer, model, scheduler, scaler, step,
                        loss_coordinate_policy="constrained", numerical_policy="inherit", sampling_policy="native",
                        anchor_policy="legacy", lambda_triangulation_anchor=0.0, supervision_policy="legacy", completion_policy="legacy", search_size_policy="shared"):
    """Keep each requested state; record enough state for a future exact resume."""
    torch.save({
        'loss_convergence_state': getattr(model, 'loss_convergence_state', None),
        **({'retracking_input_sha256': model.retracking_input_sha256}
           if supervision_policy == 'retracking_aux' else {}),
        'loss_component_contract': 'raw_final_iteration_v2',
        'semantic_identity': semantic_identity(model),
        'fnet_update_policy': getattr(model, 'fnet_update_policy', 'legacy'),
        'fnet_lr_scale': getattr(model, 'fnet_lr_scale', 1.0),
        **fnet_stage_metadata(model),
        'optimizer_parameter_groups': getattr(model, 'optimizer_parameter_groups', None),
        'trainable_parameter_names': [name for name,p in model.named_parameters() if p.requires_grad],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'optimizer_steps': step,
        'prediction_contract': 'post_update_eval_v1',
        'loss_coordinate_policy': loss_coordinate_policy,
        'sampling_policy': sampling_policy,
        'anchor_policy': anchor_policy,
        **anchor_metadata(anchor_policy),
        'search_size_policy': search_size_policy,
        'search_size_contract_json': json.dumps(SEARCH_SIZE_CONTRACT, sort_keys=True),
        'completion_policy': completion_policy,
        'completion_contract_json': json.dumps(COMPLETION_CONTRACT, sort_keys=True),
        'supervision_policy': supervision_policy,
        'supervision_contract_json': json.dumps(supervision_contract(supervision_policy), sort_keys=True),
        'lambda_triangulation_anchor': lambda_triangulation_anchor,
        'numerical_policy': numerical_policy,
        'numerical_settings': numerical_settings(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state_all': torch.cuda.get_rng_state_all(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }, os.path.join(path, f'model-{step:09d}.pth'))


def main(
        B=1, 
        V=4, 
        S=16, 
        N=128, 
        stride=4, 
        size=4, 
        threshold=0.01, 
        iters=8, 
        crop_size=(360, 640), 
        dataset_location='', 
        target_seq='000002',
        lr=5e-4,
        grad_acc=1, 
        max_iters=5000, 
        log_dir='./logs_tto',
        ckpt_dir='./checkpoints_tto',
        save_freq=500,
        vis_freq=100,  # 可视化频率
        eval_freq=100,
        pred_freq=0,
        device_ids=[0],
        init_dir='', 
        seed=125,
        sample_seed=None,
        sample_cache_dir='',
        require_sample_cache=False,
        expected_selected_views=None,
        cache_in_memory=True,
        run_name='',
        grid_dim=7,
        target_scale=0.25,
        lambda_smooth=10.0,
        lambda_rigid=0.2,
        adaptive_size=True,
        loss_on_final_only=False,
        use_dino=True,
        use_cotracker=True,
        triangulation_init_root='',
    lambda_triangulation_anchor=0.0,
    visibility_source="model",
    loss_coordinate_policy="constrained",
    numerical_policy="inherit",
    scheduler_steps=None,
    repeatability_trace=False,
    sampling_policy="native",
    anchor_policy="legacy",
    supervision_policy="legacy",
    completion_policy="legacy",
    search_size_policy="shared",
    semantic_encoder="dinov2",
    semantic_cache_path="",
    common_init_checkpoint="",
    correlation_geometry="legacy",
    dino_query_policy="legacy",
    view_weight_policy="legacy",
    fnet_update_policy="legacy",
    fnet_lr_scale=1.0,
    fnet_warmup_steps=0,
    scheduler_total_steps=None,
    resume_checkpoint_path="",
    loss_diagnostics=False,
    convergence_config_path="",
    coordinate_update_policy="legacy",
    score_fusion_policy="legacy",
    cotracker_encoder_checkpoint="",
):
    visibility_source = str(visibility_source).lower()
    if visibility_source not in ("model", "triangulation", "geometry_fused"):
        raise ValueError(f"visibility_source must be 'model', 'triangulation', or 'geometry_fused', got {visibility_source!r}")
    if B != 1 or target_seq is None:
        raise ValueError("Scene TTO exports require B=1 and one target_seq")
    if max_iters < 0 or grad_acc < 1:
        raise ValueError("max_iters must be nonnegative and grad_acc positive")
    if repeatability_trace and grad_acc != 1:
        raise ValueError('repeatability_trace currently requires grad_acc=1')
    convergence_recipe = load_recipe(convergence_config_path)
    progress_driven = convergence_recipe is not None and convergence_recipe['schedule']['kind']=='loss_plateau'
    if progress_driven and (save_freq != convergence_recipe['stopping']['window'] or pred_freq != save_freq):
        raise ValueError('Progress convergence requires save_freq=pred_freq=loss window')
    custom_schedule = convergence_recipe is not None and convergence_recipe['schedule']['kind'] != 'legacy'
    effective_scheduler_steps = resolve_schedule(0 if custom_schedule else max_iters, scheduler_steps, scheduler_total_steps)
    collect_loss_diagnostics = bool(loss_diagnostics or convergence_recipe)
    if collect_loss_diagnostics and grad_acc != 1:
        raise ValueError('Loss convergence currently requires grad_acc=1')
    validate_sampling_policy(sampling_policy)
    validate_anchor(anchor_policy, lambda_triangulation_anchor)
    validate_supervision(supervision_policy)
    validate_completion(completion_policy)
    validate_search_size(search_size_policy)
    validate_fnet_updates(fnet_update_policy)
    validate_fnet_lr(fnet_lr_scale, fnet_update_policy)
    validate_fnet_warmup(fnet_warmup_steps, fnet_update_policy)
    if fnet_warmup_steps and (not use_cotracker or grad_acc != 1):
        raise ValueError('Staged fnet warmup requires enabled fnet and grad_acc=1')
    numerics = configure_numerics(numerical_policy)
    sample_seed = seed if sample_seed is None else int(sample_seed)

    set_random_seed(seed)
    device = 'cuda:%d' % device_ids[0]
    
    base_model_name = f"TTO_{B}_{S}_{N}_lr{lr}_i{iters}"
    model_name = str(run_name) if run_name else base_model_name
    print('Starting TTO:', model_name)
    
    ckpt_path = os.path.join(ckpt_dir, model_name)
    vis_path = os.path.join(ckpt_path, "visualizations") 
    pred_path = os.path.join(ckpt_path, "predictions")
    os.makedirs(ckpt_path, exist_ok=False)
    writer = SummaryWriter(os.path.join(log_dir, model_name), flush_secs=30)
    os.makedirs(vis_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)
    run_config = {
        "data": {
            "dataset_location": dataset_location,
            "target_seq": target_seq,
            "B": B,
            "V": V,
            "S": S,
            "N": N,
            "crop_size": crop_size,
            "sample_cache_dir": sample_cache_dir,
            "cache_in_memory": cache_in_memory,
            "sample_seed": sample_seed,
            "require_sample_cache": require_sample_cache,
            "expected_selected_views": expected_selected_views,
        },
        "model": {
            "base_model_name": base_model_name,
            "run_name": run_name,
            "stride": stride,
            "initial_size": size,
            "threshold": threshold,
            "iters": iters,
            "init_dir": init_dir,
            "grid_dim": grid_dim,
            "target_scale": target_scale,
            "lambda_smooth": lambda_smooth,
            "lambda_rigid": lambda_rigid,
            "adaptive_size": adaptive_size,
            "loss_on_final_only": loss_on_final_only,
            "loss_coordinate_policy": loss_coordinate_policy,
            "sampling_policy": sampling_policy,
            "anchor_policy": anchor_policy,
            **anchor_metadata(anchor_policy),
            "search_size_policy": search_size_policy,
            "search_size_contract": SEARCH_SIZE_CONTRACT,
            "completion_policy": completion_policy,
            "completion_contract": COMPLETION_CONTRACT,
            "supervision_policy": supervision_policy,
            "supervision_contract": supervision_contract(supervision_policy),
            "use_dino": use_dino,
            "semantic_encoder": semantic_encoder,
            "semantic_cache_path": semantic_cache_path,
            "common_init_checkpoint": common_init_checkpoint,
            "correlation_geometry": correlation_geometry,
            "dino_query_policy": dino_query_policy,
            "view_weight_policy": view_weight_policy,
            "coordinate_update_policy": coordinate_update_policy,
            "score_fusion_policy": score_fusion_policy,
            "use_cotracker": use_cotracker,
            "triangulation_init_root": triangulation_init_root,
            "lambda_triangulation_anchor": lambda_triangulation_anchor,
            "visibility_source": visibility_source,
        },
        "trainer": {
            "loss_component_contract": "raw_final_iteration_v2",
            "loss_diagnostics_schema": (3 if supervision_policy == 'retracking_aux' else 2) if collect_loss_diagnostics else None,
            "convergence_recipe": convergence_recipe,
            "resume_checkpoint_path": resume_checkpoint_path,
            "fnet_update_policy": fnet_update_policy,
            "fnet_lr_scale": float(fnet_lr_scale),
            "fnet_warmup_steps": fnet_warmup_steps,
            "lr": lr,
            "grad_acc": grad_acc,
            "max_iters": max_iters,
            "scheduler_steps": effective_scheduler_steps,
            "scheduler_total_steps": max_iters if custom_schedule else effective_scheduler_steps + 100,
            "numerical_settings": numerics,
            "repeatability_trace": repeatability_trace,
            "save_freq": save_freq,
            "vis_freq": vis_freq,
            "eval_freq": eval_freq,
            "pred_freq": pred_freq,
            "log_dir": log_dir,
            "ckpt_dir": ckpt_dir,
            "device_ids": device_ids,
            "seed": seed,
            "optimization_seed": seed,
            "prediction_contract": "post_update_eval_v1",
            "prediction_mode": "eval",
            "zero_step_definition": "model forward before any optimizer update (not triangulation alone)",
            "query_visibility_source": "gt_query",
        },
    }
    save_run_config(ckpt_path, run_config)

    logger = ExperimentLogger(save_dir=ckpt_path)

    dataset = ExportDataset(
        dataset_location=dataset_location,
        V=V, S=S, N=N,
        # crop_size=crop_size,
        specific_seq=target_seq,
        seed=sample_seed,
        sample_cache_dir=sample_cache_dir or os.path.join(ckpt_path, "point_cache"),
        cache_in_memory=cache_in_memory,
        triangulation_init_root=triangulation_init_root,
        require_sample_cache=require_sample_cache,
        expected_selected_views=expected_selected_views,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=B,
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    iterloader = iter(dataloader)

    model = Pips(stride=stride, grid_dim=grid_dim, load_dino=use_dino,
                 semantic_encoder=semantic_encoder, semantic_cache_path=semantic_cache_path,
                 correlation_geometry=correlation_geometry, dino_query_policy=dino_query_policy,
                 view_weight_policy=view_weight_policy, coordinate_update_policy=coordinate_update_policy,
                 score_fusion_policy=score_fusion_policy).to(device)
    
    if init_dir:
        _ = saverloader.load(init_dir, model)
        print(f"Loaded pretrained weights from {init_dir}")
    else:
        pass

    # 假设你已经运行了 extract_weights.py 并生成了这个文件
    if cotracker_encoder_checkpoint and os.path.isfile(cotracker_encoder_checkpoint):
        encoder_params = torch.load(cotracker_encoder_checkpoint, map_location='cpu', weights_only=True)
        model.fnet.load_state_dict(encoder_params, strict=False) # 使用 strict=False 更稳健
        print("✅ CoTracker encoder weights loaded.")
    else:
        raise FileNotFoundError("A valid cotracker_encoder_checkpoint is required")

    # for param in model.fnet.parameters():
    #     param.requires_grad = False
    if getattr(model, 'dino', None) is not None:
        for param in model.dino.parameters():
            param.requires_grad = False
            
    common_identity = load_common_initialization(model, common_init_checkpoint) if common_init_checkpoint else None
    run_config['trainer']['parameter_selection'] = configure_fnet_updates(model, fnet_update_policy)
    optimizer_params, optimizer_groups = configure_fnet_optimizer(model, fnet_lr_scale)
    run_config['trainer']['optimizer_parameter_groups'] = optimizer_groups
    run_config['model']['semantic_identity'] = semantic_identity(model)
    run_config['model']['common_initialization'] = common_identity
    save_run_config(ckpt_path, run_config)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer, scheduler = fetch_optimizer(lr, 1e-5, 1e-8, effective_scheduler_steps, optimizer_params,
        lr_scales=None if float(fnet_lr_scale) == 1 else [g['lr_scale'] for g in optimizer_groups])
    configure_fnet_warmup(model, fnet_warmup_steps)
    model.collect_loss_diagnostics = collect_loss_diagnostics
    if custom_schedule:
        scheduler_class = ProgressSchedule if progress_driven else ProtocolSchedule
        scheduler = scheduler_class(optimizer, convergence_recipe['schedule'], max_iters,
            [g['lr_scale'] for g in optimizer_groups])
    monitor = ProgressMonitor(convergence_recipe) if progress_driven else (LossPlateau(convergence_recipe['stopping']) if convergence_recipe else None)

    use_amp = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"🚀 AMP Enabled. Using dtype: {amp_dtype}")
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    model.train()
    if model.dino is not None:
        model.dino.eval()

    forward_kwargs = dict(
        iters=iters, target_scale=target_scale, lambda_smooth=lambda_smooth,
        lambda_rigid=lambda_rigid, adaptive_size=adaptive_size,
        loss_on_final_only=loss_on_final_only, use_dino=use_dino,
        use_cotracker=use_cotracker, lambda_triangulation_anchor=lambda_triangulation_anchor,
        loss_coordinate_policy=loss_coordinate_policy, sampling_policy=sampling_policy,
        anchor_policy=anchor_policy, supervision_policy=supervision_policy, completion_policy=completion_policy, search_size_policy=search_size_policy,
    )
    # Loading the fixed sample separately avoids advancing the training sampler.
    export_sample = torch.utils.data.default_collate([dataset[0]])
    if supervision_policy == 'retracking_aux':
        if 'retracking_input_sha256' not in export_sample:
            raise ValueError('Missing auxiliary coordinate sidecar')
        run_config['model']['retracking_input_sha256'] = list(export_sample['retracking_input_sha256'])
        save_run_config(ckpt_path, run_config)
    start_step = 0
    if resume_checkpoint_path:
        start_step = restore_tto_state(resume_checkpoint_path, model, optimizer, scheduler, scaler,
            run_config, len(dataset), max_iters,
            restart_scheduler=bool(convergence_recipe and convergence_recipe.get('restart_schedule', False)))
        # The old one-item loader is exhausted after its saved optimizer update.
        # Recreate it on the next update, after restoring all global RNG states.
        iterloader = iter(())
        if monitor is not None and not convergence_recipe.get('restart_schedule', False):
            saved_monitor = torch.load(resume_checkpoint_path, map_location='cpu', weights_only=False).get('loss_convergence_state')
            if saved_monitor is None:
                raise ValueError('Exact convergence resume requires saved monitor state')
            monitor.load_state_dict(saved_monitor)
            if progress_driven and monitor.selected_step is not None and monitor.selected_step < start_step:
                carry_selected_checkpoint(resume_checkpoint_path, ckpt_path, monitor.selected_step)
    model.loss_convergence_state = monitor.state_dict() if monitor is not None else None
    current_vis_data = predict_for_export(
        model, export_sample, stride, device, size, threshold, amp_dtype,
        step=start_step, **forward_kwargs,
    )
    if progress_driven:
        monitor.consider_checkpoint(start_step, model.last_loss_diagnostics)
        model.loss_convergence_state = monitor.state_dict()
    save_prediction_npz(pred_path, start_step, current_vis_data, visibility_source=visibility_source,
                        numerical_policy=numerical_policy)
    save_tto_checkpoint(ckpt_path, optimizer, model, scheduler, scaler, start_step,
                        loss_coordinate_policy, numerical_policy, sampling_policy,
                        anchor_policy, lambda_triangulation_anchor, supervision_policy, completion_policy, search_size_policy)
    trace_path = os.path.join(ckpt_path, 'optimization_trace.jsonl')
    if repeatability_trace:
        write_trace(trace_path, start_step, model, optimizer, scheduler, scaler,
                    initial_eval_prediction=state_digest(current_vis_data['trajs_e_3d']))
    
    global_step = start_step
    stop_reason = 'budget_exhausted'
    
    while global_step < max_iters:
        global_step += 1
        apply_fnet_stage(model, global_step)
        iter_start_time = time.time()
        total_loss_accum = 0.0
        current_vis_data = None 

        for internal_step in range(grad_acc):
            try:
                sample = next(iterloader)
            except StopIteration:
                iterloader = iter(dataloader)
                sample = next(iterloader)
            
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss, metrics, vis_data = run_model(
                    model, sample, stride, device, size, threshold,
                    iters=iters,
                    is_train=True,
                    step=global_step,
                    target_scale=target_scale,
                    lambda_smooth=lambda_smooth,
                    lambda_rigid=lambda_rigid,
                    adaptive_size=adaptive_size,
                    loss_on_final_only=loss_on_final_only,
                    use_dino=use_dino,
                    use_cotracker=use_cotracker,
                    lambda_triangulation_anchor=lambda_triangulation_anchor,
                    loss_coordinate_policy=loss_coordinate_policy, sampling_policy=sampling_policy,
                    anchor_policy=anchor_policy, supervision_policy=supervision_policy, completion_policy=completion_policy, search_size_policy=search_size_policy,
                )
                current_vis_data = vis_data

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                raise FloatingPointError(f"Non-finite loss before optimizer update {global_step}")

            loss = loss / grad_acc
            scaler.scale(loss).backward()
            total_loss_accum += loss.item()

        scaler.unscale_(optimizer)
        unclipped_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        stop_requested = False
        if collect_loss_diagnostics:
            loss_observation = dict(step=global_step, training_loss=total_loss_accum,
                learning_rates_used=[g['lr'] for g in optimizer.param_groups],
                unclipped_grad_norm=float(unclipped_grad_norm), diagnostics=metrics['loss_diagnostics'])
            loss_observation.update(fnet_stage_metadata(model))
            with open(os.path.join(ckpt_path, 'loss_trace.jsonl'), 'a') as handle:
                handle.write(json.dumps(loss_observation, allow_nan=False)+'\n')
            if monitor is not None and global_step > fnet_warmup_steps:
                extra = {'lr':optimizer.param_groups[0]['lr']} if progress_driven else {}
                plateau = monitor.observe(global_step, total_loss_accum, metrics['loss_diagnostics'], **extra)
                model.loss_convergence_state = monitor.state_dict()
                stop_requested = plateau and convergence_recipe['stopping'].get('enabled', True)
                if stop_requested:stop_reason = 'loss_plateau'

        if not grads_are_finite(trainable_params):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite gradients before optimizer update {global_step}")

        if repeatability_trace:
            observations = dict(
                learning_rate_used=optimizer.param_groups[0]['lr'],
                training_loss=total_loss_accum,
                loss_components={key: metrics[key] for key in ('rep_loss', 'smooth_loss', 'rigid_loss', 'anchor_loss')},
                sample_indices=state_digest(vis_data['sample_indices']),
                pre_update_training_prediction=state_digest(vis_data['trajs_e_3d']),
                clipped_gradients=gradient_digest(model),
            )
            if float(fnet_lr_scale) != 1:
                observations['learning_rates_used'] = [g['lr'] for g in optimizer.param_groups]
            observations.update(fnet_stage_metadata(model))
            if 'auxiliary_loss' in metrics:
                observations['loss_components']['auxiliary_loss'] = metrics['auxiliary_loss']
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if progress_driven and monitor.requested_reduction:
            scheduler.reduce()
        if repeatability_trace:
            write_trace(trace_path, global_step, model, optimizer, scheduler, scaler, **observations)
        optimizer.zero_grad()

        needs_export = stop_requested or global_step == max_iters or any(
            frequency and global_step % frequency == 0
            for frequency in (eval_freq, vis_freq, pred_freq, save_freq)
        )
        if needs_export:
            current_vis_data = None
            del vis_data, loss
            current_vis_data = predict_for_export(
                model, export_sample, stride, device, size, threshold, amp_dtype,
                step=global_step, **forward_kwargs,
            )
            if progress_driven and (global_step % save_freq==0 or stop_requested or global_step==max_iters):
                monitor.consider_checkpoint(global_step, model.last_loss_diagnostics)
                model.loss_convergence_state = monitor.state_dict()

        time_per_iter = time.time() - iter_start_time
        
        if global_step % 10 == 0:
            print(f"Iter {global_step}/{max_iters} | "
                  f"Loss: {total_loss_accum:.4f} | "
                  f"Rep: {metrics['rep_loss']:.3f} | "
                  f"Sm: {metrics['smooth_loss']:.3f} | "
                  f"Rig: {metrics['rigid_loss']:.3f} | "
                  f"Anc: {metrics['anchor_loss']:.3f} | "
                  f"GT_L1: {metrics['l1_avg']:.3f} | "
                  f"Time: {time_per_iter:.2f}s")
            
            writer.add_scalar('Train/Total_Loss', total_loss_accum, global_step)
            writer.add_scalar('Validation/L1_Error_GT', metrics['l1_avg'], global_step)

        
        # ==========================================
        # [新增] 2D 指标评测与对比 (TAPVid Evaluator)
        # ==========================================
        if eval_freq and global_step % eval_freq == 0 and current_vis_data is not None:
            # 使用 torch.no_grad() 避免显存占用
            with torch.no_grad():
                # 提取数据
                d = current_vis_data
                eval_visibility = select_prediction_visibility(d, visibility_source)
                if eval_visibility.ndim == 3:
                    eval_visibility = eval_visibility.unsqueeze(-1).expand_as(d['vis_prob_dino'])
                
                # 调用我们在 eval_utils 中写的批量对比函数
                avg_tto, avg_co = evaluate_batch_comparison(
                    trajs_e_3d=d['trajs_e_3d'],
                    vis_prob_dino=eval_visibility,
                    cotracker_uv=d['cotracker_uv'],
                    cotracker_vis=d['cotracker_vis'],
                    trajs_g_3d=d['trajs_g_3d'],
                    vis_g_3d=d['vis_g'],
                    query_t=d['query_t'], # <--- [新增] 传入查询时间索引
                    K=d['K'], 
                    RT=d['RT'], 
                    H=d['H'], 
                    W=d['W'],
                    project_func=project_points_3d_to_2d # 传入本文件定义的投影函数
                )
                
                # 记录到 CSV 和 TXT
                logger.log_result(global_step, avg_tto, avg_co)
                
                # (可选) 写入 TensorBoard
                writer.add_scalar('Eval/AJ_Avg_Ours', avg_tto['AJ_Avg'], global_step)
                writer.add_scalar('Eval/AJ_Avg_Base', avg_co['AJ_Avg'], global_step)
                writer.add_scalar('Eval/Delta_Avg_Ours', avg_tto['Delta_Avg'], global_step)

        # --- 可视化调用 ---
        if vis_freq and global_step % vis_freq == 0 and current_vis_data is not None:
            print(f"🖌️ Generating visualization for step {global_step}...")
            try:
                # 调用新的图片可视化函数
                visualize_multi_view_images(
                    rgbs=current_vis_data['rgbs'],
                    trajs_g_3d=current_vis_data['trajs_g_3d'],
                    trajs_e_3d=current_vis_data['trajs_e_3d'],
                    vis_g=current_vis_data['vis_g'],
                    K=current_vis_data['K'],
                    RT=current_vis_data['RT'],
                    query_t=current_vis_data['query_t'], # <--- [新增] 传入 query_t
                    save_root=vis_path,
                    step=global_step
                )
            except Exception as e:
                print(f"⚠️ Vis Error: {e}")
                import traceback
                traceback.print_exc()

        pred_saved_this_step = False
        if pred_freq and global_step % pred_freq == 0 and current_vis_data is not None:
            save_prediction_npz(pred_path, global_step, current_vis_data, visibility_source=visibility_source,
                                numerical_policy=numerical_policy)
            pred_saved_this_step = True

        if (save_freq and global_step % save_freq == 0) or global_step == max_iters or stop_requested:
            save_tto_checkpoint(ckpt_path, optimizer, model, scheduler, scaler, global_step,
                                loss_coordinate_policy, numerical_policy, sampling_policy,
                                anchor_policy, lambda_triangulation_anchor, supervision_policy, completion_policy, search_size_policy)
            if current_vis_data is not None and not pred_saved_this_step:
                save_prediction_npz(pred_path, global_step, current_vis_data, visibility_source=visibility_source,
                                    numerical_policy=numerical_policy)
        if stop_requested:
            break

    print("TTO Training Finished.")
    if monitor is not None:
        with open(os.path.join(ckpt_path, 'convergence.json'), 'w') as handle:
            json.dump(monitor.report(stop_reason, global_step), handle, indent=2)
    if current_vis_data is not None:
        save_prediction_npz(pred_path, global_step, current_vis_data, final=True, visibility_source=visibility_source,
                            numerical_policy=numerical_policy)
    if progress_driven:
        if monitor.selected_step is None:
            raise RuntimeError('No checkpoint has valid supervision for loss selection')
        # Copy the already frozen prediction associated with the saved best-loss
        # checkpoint. Do not mix its model weights with the terminal optimizer.
        shutil.copyfile(os.path.join(pred_path, f'step-{monitor.selected_step:06d}_tracks.npz'),
                        os.path.join(pred_path, 'final_tracks.npz'))
    writer.close()

if __name__ == '__main__':
    Fire(main)
