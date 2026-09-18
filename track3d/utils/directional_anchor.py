"""Protect trustworthy raw geometry only in weakly supervised directions."""
import torch


CONTRACT = dict(version='raw_weak_geometry_v1', confidence_floor=0.3,
    max_reprojection_px=2.0, min_ray_angle_deg=2.0,
    weak_eigenvalue_ratio=0.01, huber_beta_pixels=1.0,
    support='post_query_raw_only_no_completion',
    rays='fixed_raw_reference_pseudo_visibility_times_original_query_visibility',
    scale='mean_positive_in_frame_camera_z_over_sqrt_fx_fy',
    normalization='all_post_query_frame_points',
    confidence_weight='original_raw_confidence_after_reliability_gate',
    application='final_iteration_loss_only', no_additional_gt=True)


def geometry(raw_tracks, raw_valid, confidence, reprojection, num_views, ray_angle,
             query_t, intrinsics, extrinsics, pseudo_visibility, query_visibility,
             image_size):
    """Build detached CPU geometry; no future GT, fill, model score or residual."""
    with torch.no_grad(), torch.amp.autocast('cpu', enabled=False):
        x, valid, conf, rep, views, angle, qt, k, rt, pv, qv = (
            t.detach().to(device='cpu', dtype=torch.float64) for t in
            (raw_tracks, raw_valid, confidence, reprojection, num_views, ray_angle,
             query_t, intrinsics, extrinsics, pseudo_visibility, query_visibility))
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError('Expected raw geometry [B,S,N,3]')
        b, s, n, _ = x.shape; v = k.shape[2]
        if (any(t.shape != (b,s,n) for t in (valid,conf,rep,views,angle))
                or qt.shape != (b,n) or k.shape != (b,s,v,3,3)
                or rt.shape != (b,s,v,4,4) or pv.shape != (b,s,v,n)
                or qv.shape != (b,1,n,v)):
            raise ValueError('Directional anchor metadata dimensions differ')
        if (not all(torch.isfinite(t).all() for t in (valid,qt,k,rt,pv,qv))
                or (qt != qt.floor()).any() or (qt < 0).any() or (qt >= s).any()
                or (pv < 0).any() or (pv > 1).any() or (qv < 0).any() or (qv > 1).any()
                or (k[...,0,0] <= 0).any() or (k[...,1,1] <= 0).any()):
            raise ValueError('Invalid camera, query or visibility input')
        h, w = image_size
        if min(h,w) <= 0: raise ValueError('Invalid image size')
        post = torch.arange(s).view(1,s,1) > qt[:,None]
        observed = (valid > .5) & post
        if (not all(torch.isfinite(t[observed]).all() for t in (x,conf,rep,views,angle))
                or (conf[observed] < 0).any() or (conf[observed] > 1).any()
                or (rep[observed] < 0).any() or (angle[observed] < 0).any()
                or (angle[observed] > 180).any()):
            raise ValueError('Invalid observed raw anchor metadata')
        x = torch.where(observed[...,None], x, torch.zeros_like(x))
        reliable = (observed & (conf > CONTRACT['confidence_floor'])
            & (rep <= CONTRACT['max_reprojection_px']) & (views >= 2)
            & (torch.minimum(angle,180-angle) >= CONTRACT['min_ray_angle_deg']))
        hom = torch.cat((x,torch.ones_like(x[...,:1])), -1)
        cam = torch.einsum('bsvij,bsnj->bsvni',rt,hom)[...,:3]
        uvh = torch.einsum('bsvij,bsvnj->bsvni',k,cam)
        positive = (cam[...,2] > 0) & (uvh[...,2] > 0) & torch.isfinite(uvh).all(-1)
        uv = uvh[...,:2] / torch.where(positive,uvh[...,2],1.)[...,None]
        in_frame = positive & (uv[...,0]>=0) & (uv[...,0]<=w-1) & (uv[...,1]>=0) & (uv[...,1]<=h-1)
        center = -torch.einsum('bsvji,bsvj->bsvi',rt[...,:3,:3],rt[...,:3,3])
        rays = x[:,:,None] - center[:,:,:,None]
        length = torch.linalg.vector_norm(rays,dim=-1,keepdim=True)
        rays = rays / torch.where(length>0,length,1.)
        support = pv * qv.permute(0,1,3,2) * in_frame
        mass = support.sum(2)
        eye = torch.eye(3,dtype=x.dtype)
        transverse = eye - rays[..., :,None]*rays[...,None,:]
        information = (support[...,None,None]*transverse).sum(2) / mass.clamp_min(1e-12)[...,None,None]
        eig, vectors = torch.linalg.eigh(information)
        maximum = eig[...,-1:]
        weakness = (1-eig.clamp_min(0)/(CONTRACT['weak_eigenvalue_ratio']*maximum.clamp_min(1e-12))).clamp(0,1)
        root = (vectors*weakness.sqrt()[...,None,:]) @ vectors.transpose(-1,-2)
        # A pixel-equivalent distance keeps the loss invariant to world-unit changes.
        local_scale = cam[...,2] / torch.sqrt(k[...,0,0]*k[...,1,1])[...,None]
        scale_weights = torch.where((mass>0)[:,:,None],support,in_frame.double())
        scale_mass = scale_weights.sum(2)
        scale = (torch.where(in_frame,local_scale,0.)*scale_weights).sum(2)/scale_mass.clamp_min(1e-12)
        scale = torch.where(scale_mass>0,scale,torch.ones_like(scale))
        weight = torch.where(reliable & (scale_mass>0),conf,0.)
        return dict(target=x.float(),root=root.float(),weight=weight.float(),scale=scale.float(),
                    weakness=weakness.float(),support=mass.float(),post=post)


def loss(prediction, fixed):
    """Radial Huber penalty in weak directions; inactive gaps have zero gradient."""
    with torch.amp.autocast(prediction.device.type, enabled=False):
        a = {k:v.to(prediction.device) for k,v in fixed.items()}
        if prediction.shape != a['target'].shape:
            raise ValueError('Prediction and directional anchor dimensions differ')
        residual = prediction.float()-a['target']
        projected = torch.einsum('bsnij,bsnj->bsni',a['root'],residual)/a['scale'][...,None]
        distance = torch.linalg.vector_norm(projected,dim=-1)
        beta = CONTRACT['huber_beta_pixels']
        penalty = torch.where(distance<beta,.5*distance.square()/beta,distance-.5*beta)
        return (penalty*a['weight']).sum()/a['post'].sum().clamp_min(1)
