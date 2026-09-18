import torch
import numpy as np
import os
import cv2
import hashlib
from pathlib import Path
from torch.utils.data import Dataset
from datasets.triangulation_metadata import observed_metadata

# ==========================================
# 辅助函数：视频/图像序列读取 (无 Resize 版)
# ==========================================

def read_video_sequence(view_folder, S, strict=True):
    """
    智能读取函数 (修改版：移除 Resize):
    1. 优先尝试读取 rgb.mp4
    2. 如果 mp4 不存在，尝试读取图片序列
    3. 自动补帧（Padding）
    4. 返回原始尺寸图像 (S, H, W, 3)
    """
    view_path = Path(view_folder)
    frames = []
    
    # 1. 尝试读取 MP4
    mp4_path = view_path / "rgb.mp4"
    if mp4_path.exists():
        cap = cv2.VideoCapture(str(mp4_path))
        count = 0
        while cap.isOpened() and count < S:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            count += 1
        cap.release()
        if strict and len(frames) != S:
            raise ValueError(f"Video {mp4_path} has {len(frames)} readable frames, expected {S}")
        
        # 补帧逻辑
        H, W = frames[0].shape[:2] if len(frames) > 0 else (0, 0)
        while len(frames) < S:
            if len(frames) == 0:
                print(f"[Error] Empty video {mp4_path}")
                break
            else:
                frames.append(frames[-1])
            
        return np.stack(frames, axis=0)

    # 2. 尝试读取图片序列
    else:
        for s in range(S):
            img_path = view_path / f'{s:06d}.png'
            if not img_path.exists():
                img_path = view_path / f'{s:06d}.jpg'
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
            else:
                img = None
            frames.append(img)
        if strict and any(frame is None for frame in frames):
            raise ValueError(f"Missing/unreadable RGB frame in {view_folder}")
        
        # 补全缺失帧
        valid_frame = None
        for f in frames:
            if f is not None:
                valid_frame = f
                break
                
        if valid_frame is None:
             print(f"[Error] No images found in {view_folder}")
             return None
             
        H, W = valid_frame.shape[:2]
        
        final_frames = []
        for i, f in enumerate(frames):
            if f is None:
                if i > 0 and final_frames[-1] is not None:
                    final_frames.append(final_frames[-1])
                else:
                    final_frames.append(np.zeros((H, W, 3), dtype=np.uint8))
            else:
                final_frames.append(f)
            
        return np.stack(final_frames, axis=0)


# ==========================================
# 核心 Dataset 类
# ==========================================

class ExportDataset(Dataset):
    def __init__(self,
                 dataset_location='',
                 V=4,      
                 S=32,     
                 N=512,    
                 crop_size=None, 
                 specific_seq=None, 
                 seed=125,
                 sample_cache_dir=None,
                 cache_in_memory=None,
                 triangulation_init_root=None,
                 require_sample_cache=False,
                 expected_selected_views=None,
    ):
        print(f'Loading dataset from {dataset_location} (Native Resolution, Fixed Indices)')
        self.dataset_location = Path(dataset_location)
        self.V = V
        self.S = S
        self.N = N
        self.seed = int(seed)
        self.require_sample_cache = require_sample_cache
        self.expected_selected_views = expected_selected_views
        self.sample_cache_dir = Path(sample_cache_dir) if sample_cache_dir else None
        self.triangulation_init_root = Path(triangulation_init_root) if triangulation_init_root else None
        if self.sample_cache_dir is not None:
            self.sample_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_in_memory = cache_in_memory
        self.sample_memory_cache = {}

        # --- [新增] 索引缓存字典 ---
        # 结构: { dataset_index: numpy_array_of_indices }
        # 用于确保 TTO 过程中每次取出的点都是同一组
        self.cached_indices = {} 

        all_folders = sorted([fn for fn in self.dataset_location.iterdir() if fn.is_dir()])

        if specific_seq is not None:
            specific_seq_name = self._resolve_sequence_name(specific_seq, all_folders)
            self.all_folder_names = [f for f in all_folders if f.name == specific_seq_name]
            if len(self.all_folder_names) == 0:
                raise ValueError(f"Sequence {specific_seq} not found in {dataset_location}")
            print(f"🔥 Single-Video TTO Mode: Optimized ONLY on {specific_seq_name}")
        else:
            self.all_folder_names = all_folders
            print(f"🌍 Dataset Adaptation Mode: Training on {len(all_folders)} sequences")
        
        print(f'Found {len(self.all_folder_names)} sequences.')
        if self.cache_in_memory is None:
            self.cache_in_memory = len(self.all_folder_names) == 1
        if self.cache_in_memory:
            print("🧠 In-memory dataset cache enabled.")
        
        # Dummy check (optional)
        if len(self.all_folder_names) > 0:
            try:
                # 简单探测一下尺寸，防止 get_fake_sample 瞎填
                view0 = self.all_folder_names[0] / "0" / "rgb.mp4"
                if view0.exists():
                    cap = cv2.VideoCapture(str(view0))
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        self.H_dummy, self.W_dummy = frame.shape[:2]
                    else:
                        self.H_dummy, self.W_dummy = 512, 896
                else:
                    self.H_dummy, self.W_dummy = 512, 896
            except:
                self.H_dummy, self.W_dummy = 512, 896

    def __getitem__(self, index):
        if self.cache_in_memory and index in self.sample_memory_cache:
            return self.sample_memory_cache[index]

        folder = self.all_folder_names[index]
        
        try:
            # 1. 读取 GT
            track_path = folder / 'track.npz'
            if not track_path.exists(): raise FileNotFoundError(f"{track_path} missing")
            d = dict(np.load(track_path, allow_pickle=True))
            
            track = d['trajs_3d']
            visibility = d['visibility']
            cam_K = d['cam_k']
            cam_RT = d['cam_rt']
            if track.ndim != 3 or track.shape[-1] != 3 or track.shape[0] < self.S:
                raise ValueError(f"Invalid/short 3D tracks: {track.shape}; requested S={self.S}")
            full_s, full_n, _ = track.shape
            if cam_K.ndim != 4 or cam_K.shape[0] != full_s or cam_K.shape[1] < self.V or cam_K.shape[-2:] != (3, 3):
                raise ValueError(f"Invalid camera intrinsics shape: {cam_K.shape}")
            if cam_RT.shape != (full_s, cam_K.shape[1], 4, 4):
                raise ValueError(f"Invalid camera extrinsics shape: {cam_RT.shape}")
            if visibility.shape != (full_s, full_n, cam_K.shape[1]):
                raise ValueError(f"Invalid visibility shape: {visibility.shape}")
            if not (np.isfinite(cam_K).all() and np.isfinite(cam_RT).all()):
                raise ValueError("Non-finite camera parameters")
            if np.any(cam_K[..., 0, 0] <= 0) or np.any(cam_K[..., 1, 1] <= 0):
                raise ValueError("Camera focal lengths must be positive")
            metric_scale = np.asarray(d.get('track_upscaling_factor', 1.0))
            if not np.isfinite(metric_scale).all() or np.any(metric_scale <= 0):
                raise ValueError("Invalid world-to-metric scale")
            if self.expected_selected_views is not None:
                if 'selected_views' not in d or not np.array_equal(d['selected_views'][:self.V], self.expected_selected_views):
                    raise ValueError("selected_views does not match the registered input protocol")
            
            if 'query_points' in d:
                query_points = d['query_points']
            else:
                N_full = track.shape[1]
                query_points = np.zeros((N_full, 4), dtype=np.float32)
                query_points[:, 1:] = track[0]
            if query_points.shape != (full_n, 4) or not np.isfinite(query_points).all():
                raise ValueError("Invalid query_points shape/values")
            if np.any(query_points[:, 0] != np.rint(query_points[:, 0])) or np.any(query_points[:, 0] < 0):
                raise ValueError("Query frame indices must be nonnegative integers")

            # 2. 读取 CoTracker
            co_path = folder / 'cotracker.npz'
            has_cotracker = False
            if co_path.exists():
                co_d = dict(np.load(co_path, allow_pickle=True))
                co_tracks_raw = co_d['pred_tracks'] 
                co_vis_raw = co_d['pred_visibility']
                has_cotracker = True
                if co_tracks_raw.shape != (cam_K.shape[1], full_s, full_n, 2) or co_vis_raw.shape != co_tracks_raw.shape[:-1]:
                    raise ValueError("Pseudo labels do not match camera/frame/point dimensions")
                if not (np.isfinite(co_tracks_raw).all() and np.isfinite(co_vis_raw).all()):
                    raise ValueError("Non-finite pseudo labels")
            else:
                raise FileNotFoundError(f"Required pseudo labels missing: {co_path}")

            # 3. 维度切片 S
            max_S = track.shape[0]
            use_S = min(self.S, max_S)
            
            track = track[:use_S]
            visibility = visibility[:use_S]
            cam_K = cam_K[:use_S]
            cam_RT = cam_RT[:use_S]
            
            if has_cotracker:
                co_tracks = co_tracks_raw[:, :use_S]
                co_vis = co_vis_raw[:, :use_S]
            else:
                co_tracks = None
                co_vis = None

            # 4. 维度切片 V
            max_V_gt = cam_K.shape[1]
            use_V = min(self.V, max_V_gt)
            
            visibility = visibility[:, :, :use_V]
            cam_K = cam_K[:, :use_V]
            cam_RT = cam_RT[:, :use_V]
            
            if has_cotracker:
                max_V_co = co_tracks.shape[0]
                use_V_co = min(use_V, max_V_co)
                co_tracks = co_tracks[:use_V_co]
                co_vis = co_vis[:use_V_co]
                
                if use_V_co < use_V:
                    pad_v = use_V - use_V_co
                    _, s_c, n_c, _ = co_tracks.shape
                    pad_tracks = np.zeros((pad_v, s_c, n_c, 2), dtype=co_tracks.dtype)
                    pad_vis = np.zeros((pad_v, s_c, n_c), dtype=co_vis.dtype)
                    co_tracks = np.concatenate([co_tracks, pad_tracks], axis=0)
                    co_vis = np.concatenate([co_vis, pad_vis], axis=0)

            # 5. 读取视频 (无 Resize)
            vrgbs_list = []
            for v in range(use_V):
                view_folder = folder / str(v)
                if not view_folder.exists():
                    raise FileNotFoundError(f"Missing RGB view: {view_folder}")
                rgbs = read_video_sequence(str(view_folder), use_S)
                vrgbs_list.append(rgbs)
            
            if not vrgbs_list: raise ValueError("No video data loaded")
            vrgbs = np.stack(vrgbs_list, axis=0) # (V, S, H, W, 3)

            # ---------------------------------------------------
            # 6. 点采样 (N) & 对齐 【核心修改：缓存机制】
            # ---------------------------------------------------
            
            # 检查是否有缓存
            if index in self.cached_indices:
                # [Hit Cache] 直接使用上次生成的索引
                inds = self.cached_indices[index]
                # print(f"Using cached indices for seq {index}") # Debug用
            else:
                cache_path = self._sample_cache_path(folder, use_S)
                if cache_path is not None and cache_path.exists():
                    inds = np.load(cache_path)["indices"]
                    print(f"🔒 Loaded cached {len(inds)} points for sequence {folder.name}")
                else:
                    if self.require_sample_cache:
                        raise FileNotFoundError(f"Required fixed point cache missing: {cache_path}")
                    # [Miss Cache] 第一次运行，进行确定性采样
                    valid_time_mask = query_points[:, 0] < use_S
                    valid_indices = np.where(valid_time_mask)[0]
                    curr_N = len(valid_indices)
                    
                    if curr_N == 0:
                        raise ValueError(f"No query points in first {use_S} frames")
                    
                    rng = np.random.default_rng(self._sequence_seed(folder.name, use_S, curr_N))
                    if curr_N > self.N:
                        sel_indices = rng.choice(valid_indices, self.N, replace=False)
                    elif curr_N < self.N:
                        diff = self.N - curr_N
                        pad_indices = rng.choice(valid_indices, diff, replace=True)
                        sel_indices = np.concatenate([valid_indices, pad_indices])
                    else:
                        sel_indices = valid_indices
                    
                    inds = sel_indices
                    if cache_path is not None:
                        np.savez_compressed(cache_path, indices=inds)
                        print(f"🔒 Wrote point cache {cache_path}")
                
                # [Write Cache] 保存索引
                self.cached_indices[index] = inds
                print(f"🔒 Cached {len(inds)} points for sequence {folder.name}")

            # 应用索引
            inds = np.asarray(inds)
            if inds.shape != (self.N,) or not np.issubdtype(inds.dtype, np.integer):
                raise ValueError(f"Point cache must contain {self.N} integer indices")
            if np.any(inds < 0) or np.any(inds >= full_n):
                raise ValueError("Point cache contains out-of-range indices")
            if np.any(query_points[inds, 0] >= use_S):
                raise ValueError("Cached query time is outside requested frame range")
            track = track[:, inds]
            visibility = visibility[:, inds]
            query_points = query_points[inds]
            
            if has_cotracker:
                co_tracks = co_tracks[:, :, inds, :]
                co_vis = co_vis[:, :, inds]
            else:
                co_tracks = np.zeros((use_V, use_S, self.N, 2), dtype=np.float32)
                co_vis = np.zeros((use_V, use_S, self.N), dtype=np.float32)

            init_tracks = None
            init_confidence = None
            init_size = None
            init_visibility = None
            init_reproj_error = None
            init_num_views = None
            init_ray_angle = None
            init_raw_valid = None
            if self.triangulation_init_root is not None:
                (
                    init_tracks,
                    init_confidence,
                    init_size,
                    init_visibility,
                    init_reproj_error,
                    init_num_views,
                    init_ray_angle,
                    init_raw_valid,
                    explicit_raw,
                ) = self._read_triangulation_init(
                    folder.name, inds, use_S, query_points
                )

            # 7. 转 Tensor
            vrgbs = vrgbs[..., ::-1].copy() 
            vrgbs = vrgbs.transpose(0, 1, 4, 2, 3) 

            sample = {
                'rgbs': torch.from_numpy(vrgbs),                 
                'track_g': torch.from_numpy(track),              
                'query_points': torch.from_numpy(query_points),  
                'sample_indices': torch.from_numpy(inds.astype(np.int64)),
                'intrinsics': torch.from_numpy(cam_K),           
                'extrinsics': torch.from_numpy(cam_RT),          
                'visibility': torch.from_numpy(visibility),      
                'cotracker_tracks': torch.from_numpy(co_tracks), 
                'cotracker_vis': torch.from_numpy(co_vis),       
            }
            if init_tracks is not None:
                sample['init_tracks_3d'] = torch.from_numpy(init_tracks.astype(np.float32))
                sample['init_confidence'] = torch.from_numpy(init_confidence.astype(np.float32))
                sample['init_size'] = torch.from_numpy(init_size.astype(np.float32))
                sample['init_visibility'] = torch.from_numpy(init_visibility.astype(np.float32))
                sample['init_reproj_error'] = torch.from_numpy(init_reproj_error.astype(np.float32))
                sample['init_num_views'] = torch.from_numpy(init_num_views.astype(np.float32))
                sample['init_ray_angle'] = torch.from_numpy(init_ray_angle.astype(np.float32))
                sample['init_raw_valid'] = torch.from_numpy(init_raw_valid.astype(np.float32))
                raw_metadata = observed_metadata(
                    init_tracks, init_raw_valid, init_confidence, init_reproj_error,
                    init_num_views, init_ray_angle, query_points, cam_K, cam_RT,
                    co_tracks, co_vis, explicit_raw=explicit_raw,
                )
                sample.update({key: torch.from_numpy(value) for key, value in raw_metadata.items()})
                sample['init_raw_source'] = 'explicit' if explicit_raw is not None else 'legacy_post_query_geometry_checked'
                # Optional independent coordinate evidence; never replaces pseudo visibility.
                auxiliary_path = self.triangulation_init_root / folder.name / 'retracking_candidates.npz'
                if auxiliary_path.is_file():
                    from track3d.utils.retracking_auxiliary import load_candidates
                    sample.update(load_candidates(auxiliary_path, sample))
            if self.cache_in_memory:
                self.sample_memory_cache[index] = sample
                rgbs_mb = sample['rgbs'].numel() * sample['rgbs'].element_size() / (1024 ** 2)
                print(f"🧠 Cached sequence {folder.name} in memory ({rgbs_mb:.1f} MB RGB tensor).")
            return sample

        except Exception as e:
            raise RuntimeError(f"Failed loading {folder}: {e}") from e

    def get_fake_sample(self):
        return {
            'rgbs': torch.zeros((self.V, self.S, 3, self.H_dummy, self.W_dummy), dtype=torch.uint8),
            'track_g': torch.zeros((self.S, self.N, 3), dtype=torch.float32),
            'query_points': torch.zeros((self.N, 4), dtype=torch.float32),
            'sample_indices': torch.arange(self.N, dtype=torch.int64),
            'intrinsics': torch.zeros((self.S, self.V, 3, 3), dtype=torch.float32),
            'extrinsics': torch.zeros((self.S, self.V, 4, 4), dtype=torch.float32),
            'visibility': torch.zeros((self.S, self.N, self.V), dtype=torch.float32),
            'cotracker_tracks': torch.zeros((self.V, self.S, self.N, 2), dtype=torch.float32),
            'cotracker_vis': torch.zeros((self.V, self.S, self.N), dtype=torch.float32),
        }

    def __len__(self):
        return len(self.all_folder_names)

    def _sequence_seed(self, scene_name, use_S, valid_count):
        key = f"{self.seed}:{scene_name}:S{use_S}:N{self.N}:valid{valid_count}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _sample_cache_path(self, folder, use_S):
        if self.sample_cache_dir is None:
            return None
        name = f"{folder.name}_S{use_S}_N{self.N}_seed{self.seed}.npz"
        return self.sample_cache_dir / name

    def _resolve_sequence_name(self, specific_seq, all_folders):
        names = {folder.name for folder in all_folders}
        seq = str(specific_seq)
        if seq in names:
            return seq
        if seq.isdigit():
            numeric = int(seq)
            for width in sorted({len(name) for name in names if name.isdigit()}, reverse=True):
                candidate = f"{numeric:0{width}d}"
                if candidate in names:
                    return candidate
        return seq

    def _read_triangulation_init(self, scene_name, inds, use_S, query_points):
        candidates = [
            self.triangulation_init_root / scene_name / "predictions" / "final_tracks.npz",
            self.triangulation_init_root / scene_name / "final_tracks.npz",
            self.triangulation_init_root / f"{scene_name}.npz",
        ]
        init_path = next((path for path in candidates if path.exists()), None)
        if init_path is None:
            raise FileNotFoundError(f"No triangulation init found for {scene_name} under {self.triangulation_init_root}")

        data = np.load(init_path, allow_pickle=True)
        init_tracks = self._read_first_key(
            data,
            ("pred_tracks_3d", "pred_trajectories_3d", "tracks_3d", "trajs_3d", "pred_tracks"),
        ).astype(np.float32)
        init_tracks = self._squeeze_batch(init_tracks)
        if init_tracks.ndim != 3 or init_tracks.shape[-1] != 3 or init_tracks.shape[0] < use_S:
            raise ValueError(f"Invalid/short triangulation init: {init_tracks.shape}")
        init_confidence = self._read_first_key(
            data,
            ("triangulation_confidence", "confidence", "pred_visibility", "visibility"),
            default=None,
        )
        init_visibility = self._read_first_key(
            data,
            ("pred_visibility", "visibility", "pred_visibilities_any_view", "vis", "vis_e"),
            default=None,
        )
        init_size = self._read_first_key(data, ("init_size", "point_size"), default=None)
        init_reproj_error = self._read_first_key(data, ("reprojection_error_px", "reproj_error_px"), default=None)
        init_num_views = self._read_first_key(data, ("num_views_used", "num_views"), default=None)
        init_ray_angle = self._read_first_key(data, ("ray_angle_deg", "ray_angle"), default=None)
        init_raw_valid = self._read_first_key(data, ("raw_triangulated_mask", "raw_valid"), default=None)
        if init_confidence is None:
            init_confidence = np.ones(init_tracks.shape[:2], dtype=np.float32)
        else:
            init_confidence = self._squeeze_batch(init_confidence).astype(np.float32)
            if init_confidence.ndim == 3:
                init_confidence = init_confidence.max(axis=-1)
        if init_visibility is None:
            init_visibility = (init_confidence > 0).astype(np.float32)
        else:
            init_visibility = self._squeeze_batch(init_visibility).astype(np.float32)
            if init_visibility.ndim == 3:
                init_visibility = init_visibility.max(axis=-1)
        if init_confidence.shape != init_tracks.shape[:2] or init_visibility.shape != init_tracks.shape[:2]:
            raise ValueError("Init confidence/visibility shape differs from coordinates")
        init_reproj_error = self._normalize_init_scalar(
            init_reproj_error, init_tracks.shape[:2], fill_value=np.nan
        )
        init_num_views = self._normalize_init_scalar(init_num_views, init_tracks.shape[:2], fill_value=0.0)
        init_ray_angle = self._normalize_init_scalar(init_ray_angle, init_tracks.shape[:2], fill_value=0.0)
        init_raw_valid = self._normalize_init_scalar(init_raw_valid, init_tracks.shape[:2], fill_value=0.0)

        init_sample_indices = None
        if "sample_indices" in data:
            init_sample_indices = np.asarray(data["sample_indices"]).reshape(-1)
            if not np.issubdtype(init_sample_indices.dtype, np.integer):
                raise ValueError("Triangulation sample_indices must be integers")

        order = self._align_init_indices(init_sample_indices, inds, init_tracks.shape[1])
        explicit_raw = None
        if 'raw_tracks_3d' in data:
            explicit_raw = self._squeeze_batch(data['raw_tracks_3d'])[:use_S, order].astype(np.float32)
        init_tracks = init_tracks[:, order]
        init_confidence = init_confidence[:, order]
        init_visibility = init_visibility[:, order]
        init_reproj_error = init_reproj_error[:, order]
        init_num_views = init_num_views[:, order]
        init_ray_angle = init_ray_angle[:, order]
        init_raw_valid = init_raw_valid[:, order]
        if init_size is not None:
            init_size = np.asarray(init_size)[order]
        if init_tracks.ndim != 3 or init_tracks.shape[-1] != 3 or init_tracks.shape[0] < use_S:
            raise ValueError(f"Invalid/short triangulation init: {init_tracks.shape}")
        if not np.isfinite(init_tracks).all() or not np.isfinite(init_confidence).all():
            raise ValueError("Non-finite filled triangulation init/confidence")

        init_tracks = init_tracks[:use_S]
        init_confidence = init_confidence[:use_S]
        init_visibility = init_visibility[:use_S]
        init_reproj_error = init_reproj_error[:use_S]
        init_num_views = init_num_views[:use_S]
        init_ray_angle = init_ray_angle[:use_S]
        init_raw_valid = init_raw_valid[:use_S]
        init_size = self._normalize_init_size(init_size, init_tracks.shape[1])
        if init_tracks.shape[0] < use_S:
            pad_s = use_S - init_tracks.shape[0]
            init_tracks = np.concatenate([init_tracks, np.repeat(init_tracks[-1:], pad_s, axis=0)], axis=0)
            init_confidence = np.concatenate([init_confidence, np.zeros((pad_s, init_confidence.shape[1]), dtype=np.float32)], axis=0)
            init_visibility = np.concatenate([init_visibility, np.zeros((pad_s, init_visibility.shape[1]), dtype=np.float32)], axis=0)
            init_reproj_error = np.concatenate([init_reproj_error, np.full((pad_s, init_reproj_error.shape[1]), np.nan, dtype=np.float32)], axis=0)
            init_num_views = np.concatenate([init_num_views, np.zeros((pad_s, init_num_views.shape[1]), dtype=np.float32)], axis=0)
            init_ray_angle = np.concatenate([init_ray_angle, np.zeros((pad_s, init_ray_angle.shape[1]), dtype=np.float32)], axis=0)
            init_raw_valid = np.concatenate([init_raw_valid, np.zeros((pad_s, init_raw_valid.shape[1]), dtype=np.float32)], axis=0)

        frames = np.arange(use_S)[:, None]
        query_t = np.clip(np.rint(query_points[:, 0]).astype(np.int64), 0, use_S - 1)
        query_xyz = query_points[:, 1:].astype(np.float32)
        pre_or_query = frames <= query_t[None, :]
        init_tracks = np.where(pre_or_query[..., None], query_xyz[None, :, :], init_tracks)
        init_confidence = np.where(pre_or_query, 0.0, init_confidence)
        init_visibility = np.where(pre_or_query, 0.0, init_visibility)
        init_reproj_error = np.where(pre_or_query, np.nan, init_reproj_error)
        init_num_views = np.where(pre_or_query, 0.0, init_num_views)
        init_ray_angle = np.where(pre_or_query, 0.0, init_ray_angle)
        init_raw_valid = np.where(pre_or_query, 0.0, init_raw_valid)

        print(f"📐 Loaded triangulation init for {scene_name}: {init_path}")
        return (
            init_tracks.astype(np.float32),
            np.clip(init_confidence, 0.0, 1.0).astype(np.float32),
            init_size.astype(np.float32),
            (init_visibility > 0.5).astype(np.float32),
            init_reproj_error.astype(np.float32),
            init_num_views.astype(np.float32),
            init_ray_angle.astype(np.float32),
            (init_raw_valid > 0.5).astype(np.float32),
            explicit_raw,
        )

    @staticmethod
    def _align_init_indices(source, requested, num_points):
        # Equal point counts are not evidence of identity. Unlabelled arrays
        # must be regenerated with explicit sample_indices, not guessed.
        if source is None or len(source) != num_points:
            raise ValueError("Triangulation init requires matching sample_indices metadata")
        source, requested = np.asarray(source), np.asarray(requested)
        if np.array_equal(source, requested):
            return np.arange(num_points)
        if len(np.unique(source)) != len(source):
            raise ValueError("Ambiguous duplicate triangulation sample_indices")
        lookup = {int(identity): i for i, identity in enumerate(source)}
        if any(int(identity) not in lookup for identity in requested):
            raise ValueError("Requested sample_indices missing from triangulation init")
        return np.array([lookup[int(identity)] for identity in requested], dtype=np.int64)

    @staticmethod
    def _normalize_init_size(init_size, num_points):
        if init_size is None:
            return np.ones((num_points, 3), dtype=np.float32)
        init_size = np.asarray(init_size, dtype=np.float32)
        if init_size.ndim == 1:
            init_size = np.repeat(init_size[:, None], 3, axis=1)
        if init_size.ndim == 2 and init_size.shape[1] == 1:
            init_size = np.repeat(init_size, 3, axis=1)
        if init_size.shape != (num_points, 3):
            raise ValueError(f"init_size shape {init_size.shape} does not match ({num_points}, 3)")
        return init_size

    @staticmethod
    def _normalize_init_scalar(values, shape, fill_value):
        if values is None:
            return np.full(shape, fill_value, dtype=np.float32)
        values = ExportDataset._squeeze_batch(values).astype(np.float32)
        if values.ndim == 3:
            values = values.max(axis=-1)
        if values.shape != shape:
            raise ValueError(f"init scalar shape {values.shape} does not match {shape}")
        return values

    @staticmethod
    def _read_first_key(npz, keys, default=None):
        for key in keys:
            if key in npz:
                return npz[key]
        return default

    @staticmethod
    def _squeeze_batch(array):
        array = np.asarray(array)
        if array.ndim >= 4 and array.shape[0] == 1:
            return array[0]
        return array
