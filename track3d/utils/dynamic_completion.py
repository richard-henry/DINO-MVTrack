"""Input-only gap completion with fixed observations and a quadratic ray fit."""
import numpy as np


COMPLETION_CONTRACT = dict(
    version='ray_motion_v1', velocity_window=6, velocity_anchors=4,
    tail_decay_frames=6.0, prior_weight=0.1, acceleration_weight=0.25,
    ray_weight=1.0, view_normalization='per_frame_support_count',
    anchors='query_and_all_post_query_raw_fixed',
    observations='pseudo_visible_query_visible_in_image',
    fallback='motion_prior_if_solve_nonfinite_or_observed_ray_depth_nonpositive',
    scope='offline_initialization_only_no_visibility_or_confidence_updates',
)


def validate_completion(policy):
    if policy not in ('legacy', 'ray_motion'):
        raise ValueError('Unknown completion_policy: ' + str(policy))


def complete_tracks(initial, raw_valid, query_points, intrinsics, extrinsics,
                    pseudo_tracks, pseudo_visibility, query_visibility, image_size):
    """Complete only post-query gaps; world-coordinate units cancel in the fit.

    Shapes: initial[S,N,3], raw[S,N], query[N,4], K/RT[S,V,...],
    pseudo[S,V,N,2], pseudo_vis[S,V,N], query_vis[V,N]. The query visibility
    is the existing input protocol, not visibility along the GT trajectory.
    This routine accepts no full GT tracks or per-frame GT visibility.
    """
    original = np.asarray(initial)
    x, q, k, rt, uv, vis, qvis = [np.asarray(a, dtype=np.float64) for a in
        (initial, query_points, intrinsics, extrinsics, pseudo_tracks,
         pseudo_visibility, query_visibility)]
    raw = np.asarray(raw_valid) > .5
    if x.ndim != 3 or x.shape[-1] != 3 or uv.ndim != 4 or uv.shape[-1] != 2:
        raise ValueError('Expected tracks[S,N,3] and pseudo[S,V,N,2]')
    s, n, _ = x.shape
    v = uv.shape[1]
    if (raw.shape != (s,n) or q.shape != (n,4) or k.shape != (s,v,3,3)
            or rt.shape != (s,v,4,4) or uv.shape != (s,v,n,2)
            or vis.shape != (s,v,n) or qvis.shape != (v,n)):
        raise ValueError('Mismatched completion input dimensions')
    if not all(np.isfinite(a).all() for a in (x,q,k,rt,uv,vis,qvis,np.asarray(raw_valid))):
        raise ValueError('Nonfinite completion input')
    if np.any(q[:,0] != np.rint(q[:,0])) or np.any(q[:,0] < 0) or np.any(q[:,0] >= s):
        raise ValueError('Invalid query times')
    h, w = image_size
    if h <= 0 or w <= 0:
        raise ValueError('Invalid image size')
    qt = q[:,0].astype(int)
    before = np.arange(s)[:,None] <= qt[None,:]
    expected = np.broadcast_to(q[None,:,1:],x.shape)
    if not np.array_equal(x[before],expected[before]):
        raise ValueError('Initial tracks must already fix pre-query/query positions')
    gap = ~before & ~raw
    prior = x.copy()
    for point in range(n):
        anchors = np.r_[qt[point],np.flatnonzero(raw[:,point] & ~before[:,point])]
        for dim in range(3):
            prior[qt[point]:,point,dim] = np.interp(
                np.arange(qt[point],s),anchors,x[anchors,point,dim])
        recent = anchors[anchors >= anchors[-1]-COMPLETION_CONTRACT['velocity_window']]
        recent = recent[-COMPLETION_CONTRACT['velocity_anchors']:]
        velocity = np.zeros(3)
        if len(recent) >= 2:
            velocity = np.median(np.diff(x[recent,point],axis=0)/np.diff(recent)[:,None],axis=0)
        tail = np.arange(anchors[-1]+1,s)
        tau = COMPLETION_CONTRACT['tail_decay_frames']
        distance = tau * (-np.expm1(-(tail-anchors[-1])/tau))
        prior[tail,point] = x[anchors[-1],point] + distance[:,None]*velocity

    inverse = np.linalg.inv(k)
    centers = -np.einsum('svji,svj->svi',rt[:,:,:3,:3],rt[:,:,:3,3])
    hom = np.concatenate([uv,np.ones((s,v,n,1))],axis=-1)
    local = np.einsum('svij,svnj->svni',inverse,hom)
    rays = np.einsum('svji,svnj->svni',rt[:,:,:3,:3],local)
    rays /= np.maximum(np.linalg.norm(rays,axis=-1,keepdims=True),np.finfo(float).tiny)
    available = ((vis>.5) & (qvis[None]>.5) & gap[:,None,:]
                 & (uv[...,0]>=0) & (uv[...,0]<w) & (uv[...,1]>=0) & (uv[...,1]<h))
    result = original.copy()
    fallback = np.zeros(n,dtype=bool)
    identity = np.eye(3)
    for point in range(n):
        missing = np.flatnonzero(gap[:,point])
        if not len(missing):
            continue
        index = {int(t): i for i,t in enumerate(missing)}
        rows = [np.eye(3*len(missing))*np.sqrt(COMPLETION_CONTRACT['prior_weight'])]
        rhs = [prior[missing,point].reshape(-1)*np.sqrt(COMPLETION_CONTRACT['prior_weight'])]
        for t in missing:
            views = np.flatnonzero(available[t,:,point])
            for view in views:
                ray = rays[t,view,point]
                projection = identity - np.outer(ray,ray)
                factor = np.sqrt(COMPLETION_CONTRACT['ray_weight']/len(views))
                a = np.zeros((3,3*len(missing)))
                a[:,3*index[t]:3*index[t]+3] = projection*factor
                rows.append(a)
                rhs.append(projection @ centers[t,view] * factor)
        for t in range(qt[point]+1,s-1):
            if not gap[t-1:t+2,point].any():
                continue
            a = np.zeros((3,3*len(missing)))
            b = np.zeros(3)
            for frame,coefficient in ((t-1,1.),(t,-2.),(t+1,1.)):
                if frame in index:
                    a[:,3*index[frame]:3*index[frame]+3] += identity*coefficient
                else:
                    b -= coefficient*x[frame,point]
            factor = np.sqrt(COMPLETION_CONTRACT['acceleration_weight'])
            rows.append(a*factor)
            rhs.append(b*factor)
        solution = np.linalg.lstsq(np.vstack(rows),np.concatenate(rhs),rcond=None)[0].reshape(-1,3)
        valid = np.isfinite(solution).all()
        for i,t in enumerate(missing):
            for view in np.flatnonzero(available[t,:,point]):
                valid = valid and np.dot(solution[i]-centers[t,view],rays[t,view,point])>0
        if valid:
            result[missing,point] = solution
        else:
            result[missing,point] = prior[missing,point]
            fallback[point] = True
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite completed tracks')
    np.testing.assert_array_equal(result[~gap],original[~gap])
    return dict(tracks=result, motion_prior=prior.astype(original.dtype),
                gap_mask=gap, supporting_views=available.sum(1).astype(np.int16),
                fallback_points=fallback)
