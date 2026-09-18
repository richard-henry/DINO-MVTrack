import os
import csv
import time
import torch
import numpy as np
from prettytable import PrettyTable

# ==========================================
# 1. 核心评估器 (TAPVidEvaluator)
# ==========================================
class TAPVidEvaluator:
    def __init__(self):
        self.thresholds = [1, 2, 4, 8, 16] 

    def compute_metrics(self, pred_uv, pred_vis, gt_uv, gt_vis, H, W, eval_mask=None):
        """
        计算各项指标，支持时间掩码。
        Args:
            eval_mask: (B, S, N) Boolean Tensor. True 表示该点该时刻有效（需要评估）。
        """
        scale_h = 256.0 / H
        scale_w = 256.0 / W
        
        pred_scaled = pred_uv.clone()
        gt_scaled = gt_uv.clone()
        pred_scaled[..., 0] *= scale_w
        pred_scaled[..., 1] *= scale_h
        gt_scaled[..., 0]   *= scale_w
        gt_scaled[..., 1]   *= scale_h

        dist_scaled = torch.norm(pred_scaled - gt_scaled, p=2, dim=-1)
        dist_orig = torch.norm(pred_uv - gt_uv, p=2, dim=-1)

        pred_vis_bool = (pred_vis > 0.5)
        gt_vis_bool   = (gt_vis > 0.5)

        # 如果没有提供 mask，默认全部有效
        if eval_mask is None:
            eval_mask = torch.ones_like(gt_vis_bool)
        
        # 确保 mask 维度匹配 (B, S, N) -> (B, S, N) 
        # 注意：输入的 pred/gt 可能是 (B, S, N) (单视角)
        
        metrics = {}
        jaccard_list = []
        delta_list = []
        
        for thr in self.thresholds:
            within_dist = (dist_scaled < thr)
            
            # --- Jaccard (AJ) 计算 ---
            # 分子分母都必须在 eval_mask 范围内
            tp = (gt_vis_bool & pred_vis_bool & within_dist & eval_mask).float().sum()
            fp = (pred_vis_bool & (~gt_vis_bool | ~within_dist) & eval_mask).float().sum()
            fn = (gt_vis_bool & ~pred_vis_bool & eval_mask).float().sum()
            
            jaccard = tp / (tp + fp + fn + 1e-8)
            jaccard_list.append(jaccard.item())
            metrics[f'AJ_{thr}'] = jaccard.item()
            
            # --- Position Accuracy (Delta) 计算 ---
            # visible_pts: GT可见 且 在评估时间窗口内
            visible_pts = (gt_vis_bool & eval_mask).float().sum() + 1e-8
            correct_pts = (gt_vis_bool & within_dist & eval_mask).float().sum()
            
            acc = correct_pts / visible_pts
            delta_list.append(acc.item())
            metrics[f'Delta_{thr}'] = acc.item()

        metrics['AJ_Avg'] = np.mean(jaccard_list)
        metrics['Delta_Avg'] = np.mean(delta_list)

        # --- OA (Overall Accuracy) ---
        # 只统计 mask 内的准确率
        correct_class = ((pred_vis_bool == gt_vis_bool) & eval_mask).float().sum()
        valid_total = eval_mask.float().sum() + 1e-8
        metrics['OA'] = (correct_class / valid_total).item()
        
        # --- MTE (Mean Trajectory Error) ---
        # 只计算 GT 可见 且 在评估窗口内 的点
        mask_mte = gt_vis_bool & eval_mask
        mte = (dist_orig * mask_mte.float()).sum() / (mask_mte.float().sum() + 1e-8)
        metrics['MTE'] = mte.item()

        return metrics

# ==========================================
# 2. 日志记录器 (ExperimentLogger)
# ==========================================
class ExperimentLogger:
    def __init__(self, save_dir, filename="eval_results.csv"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.csv_path = os.path.join(save_dir, filename)
        self.txt_path = os.path.join(save_dir, filename.replace('.csv', '.txt'))
        
        self.metrics_keys = ['AJ_Avg'] + [f'AJ_{t}' for t in [1,2,4,8,16]] + \
                            ['Delta_Avg'] + [f'Delta_{t}' for t in [1,2,4,8,16]] + \
                            ['OA', 'MTE']
        self.headers = ['Step', 'Method'] + self.metrics_keys
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                csv.writer(f).writerow(self.headers)

    def log_result(self, step, avg_tto, avg_co):
        row_base = [step, "CoTracker"] + [f"{avg_co.get(k, 0):.4f}" for k in self.metrics_keys]
        row_ours = [step, "TTO(Ours)"] + [f"{avg_tto.get(k, 0):.4f}" for k in self.metrics_keys]
        gains = [avg_tto.get(k, 0) - avg_co.get(k, 0) for k in self.metrics_keys]
        row_gain = [step, "Gain"] + [f"{g:+.4f}" for g in gains]

        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_base)
            writer.writerow(row_ours)
            writer.writerow(row_gain)
            writer.writerow([]) 

        table = PrettyTable()
        table.field_names = ["Metric", "CoTracker", "TTO", "Gain"]
        table.align = "l"
        for k, b, m, g_str in zip(self.metrics_keys, row_base[2:], row_ours[2:], row_gain[2:]):
            g_val = float(g_str)
            color_g = f"\033[92m{g_str}\033[0m" if (g_val > 0 and k!='MTE') or (g_val < 0 and k=='MTE') else (f"\033[91m{g_str}\033[0m" if g_val != 0 else g_str)
            table.add_row([k, b, m, color_g])

        print(f"\n📊 Eval Step {step}:")
        print(table)
        
        with open(self.txt_path, mode='a') as f:
            f.write(f"\n=== Step {step} ===\n")
            f.write(table.get_string())

# ==========================================
# 3. 批量对比函数 (Helper Function)
# ==========================================
def evaluate_batch_comparison(trajs_e_3d, vis_prob_dino, cotracker_uv, cotracker_vis, trajs_g_3d, vis_g_3d, query_t, K, RT, H, W, project_func):
    """
    Project 3D -> 2D, then compare Ours vs CoTracker
    [新增参数] query_t: (B, N) 查询时间索引，用于生成评估掩码
    """
    evaluator = TAPVidEvaluator()
    V = K.shape[2]
    metrics_tto, metrics_co = [], []
    
    # ==========================================
    # [新增] 构造时间评估掩码 (Valid Time Mask)
    # ==========================================
    # 逻辑：只评估 t >= query_t 的帧 (即 Query Frame 及其之后的帧)
    # 忽略 t < query_t 的帧，因为那里的 GT 可能被简单置0，且 CoTracker 是双向生成但在 Inference 时可能是单向的
    
    B, S, N, _ = trajs_e_3d.shape
    time_indices = torch.arange(S, device=query_t.device).reshape(1, S, 1) # (1, S, 1)
    q_t_expanded = query_t.unsqueeze(1) # (B, 1, N)
    
    # eval_mask: (B, S, N)
    # True = Valid (需要评估), False = Invalid (忽略)
    valid_time_mask = (time_indices >= q_t_expanded) 
    
    for v in range(V):
        # ... Projection Logic ...
        gt_uv = project_func(trajs_g_3d, K[:,:,v], RT[:,:,v])
        pred_uv = project_func(trajs_e_3d, K[:,:,v], RT[:,:,v])
        
        # Visibility Logic
        in_frame = (pred_uv[...,0] >= 0) & (pred_uv[...,0] < W) & (pred_uv[...,1] >= 0) & (pred_uv[...,1] < H)
        pred_vis = (vis_prob_dino[..., v] > 0.5) & in_frame
        
        gt_vis_v = vis_g_3d[..., v] > 0.5
        
        # CoTracker Logic
        co_uv = cotracker_uv[:, :, v] 
        co_vis = cotracker_vis[:, :, v] > 0.5
        
        # [关键修改] 将 valid_time_mask 传入 compute_metrics
        # valid_time_mask 是 (B, S, N)，在计算时会自动广播或直接匹配
        metrics_tto.append(evaluator.compute_metrics(pred_uv, pred_vis, gt_uv, gt_vis_v, H, W, eval_mask=valid_time_mask))
        metrics_co.append(evaluator.compute_metrics(co_uv, co_vis, gt_uv, gt_vis_v, H, W, eval_mask=valid_time_mask))
        
    avg_tto = {k: np.mean([m[k] for m in metrics_tto]) for k in metrics_tto[0]}
    avg_co  = {k: np.mean([m[k] for m in metrics_co])  for k in metrics_co[0]}

    # ==========================================
    # [新增] 详细可见性诊断 (Visibility Diagnostics)
    # ==========================================
    
    vis_e0 = vis_g_3d[:, 0:1, :, :] # (B, 1, N, V)
    
    # 扩展 mask 以匹配多视角统计: (B, S, N) -> (B, S, N, 1) -> (B, S, N, V)
    # 这样统计混淆矩阵时，也会忽略 Query 之前的脏数据
    vis_stats_mask = valid_time_mask.unsqueeze(-1).expand(-1, -1, -1, V)
    
    # 计算统计 (传入 eval_mask)
    vis_stats = compute_detailed_vis_stats(vis_prob_dino, vis_g_3d, vis_e0, threshold=0.5, eval_mask=vis_stats_mask)
    
    table = PrettyTable()
    table.field_names = ["Condition", "Acc", "Prec", "Recall", "FPR (误报率)", "TP", "FP", "TN", "FN"]
    table.float_format = ".4"
    
    print("\n🔍 [Visibility Confusion Matrix Analysis (t >= query_t)]")
    for s in vis_stats:
        fpr_str = f"{s['FPR']:.4f}"
        if "Invisible" in s['Type'] and s['FPR'] > 0.1:
            fpr_str = f"\033[91m{fpr_str} (High!)\033[0m"
            
        table.add_row([
            s['Type'], s['Acc'], s['Prec'], s['Recall'], fpr_str,
            s['TP'], s['FP'], s['TN'], s['FN']
        ])
    
    print(table)
    print("-" * 60)
    
    return avg_tto, avg_co

def compute_detailed_vis_stats(pred_vis, gt_vis, query_vis_mask, threshold=0.5, eval_mask=None):
    """
    统计可见性指标，并根据 Query Frame 的可见性进行拆分。
    [新增] 支持 eval_mask，只统计有效帧
    """
    # 1. 基础二值化
    pred_bool = (pred_vis > threshold)
    gt_bool   = (gt_vis > 0.5)
    
    # 2. 如果提供了评估掩码，先过滤
    if eval_mask is not None:
        # 确保掩码广播正确
        valid = eval_mask.expand_as(gt_vis)
    else:
        valid = torch.ones_like(gt_vis, dtype=torch.bool)

    # 3. 计算混淆矩阵 (只考虑 Valid 区域)
    # TP: 预测可见 & 真值可见 & 在评估期内
    is_tp = pred_bool & gt_bool & valid
    # FP: 预测可见 & 真值不可见 & 在评估期内
    is_fp = pred_bool & (~gt_bool) & valid
    # TN: 预测不可见 & 真值不可见 & 在评估期内
    is_tn = (~pred_bool) & (~gt_bool) & valid
    # FN: 预测不可见 & 真值可见 & 在评估期内
    is_fn = (~pred_bool) & gt_bool & valid
    
    # 4. 构造分组 Mask
    mask_query_visible = query_vis_mask.expand_as(gt_vis) > 0.5
    mask_query_invisible = ~mask_query_visible
    
    # 5. 统计辅助函数
    def get_stats(mask, name):
        # 这里的 mask 只是 Query Visible/Invisible 的分组
        # 时间维度的有效性已经在 is_tp/is_fp 等计算中包含了
        
        tp = (is_tp & mask).float().sum().item()
        fp = (is_fp & mask).float().sum().item()
        tn = (is_tn & mask).float().sum().item()
        fn = (is_fn & mask).float().sum().item()
        
        total = tp + fp + tn + fn + 1e-8
        
        accuracy  = (tp + tn) / total
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8) 
        fpr       = fp / (fp + tn + 1e-8) 
        
        return {
            "Type": name,
            "Acc": accuracy,
            "Prec": precision,
            "Recall": recall,
            "FPR": fpr,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn)
        }

    stats_visible   = get_stats(mask_query_visible, "Query Visible (Easy)")
    stats_invisible = get_stats(mask_query_invisible, "Query Invisible (Hard)")
    stats_all       = get_stats(torch.ones_like(gt_vis, dtype=torch.bool), "Overall")
    
    return [stats_all, stats_visible, stats_invisible]