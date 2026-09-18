"""Frozen multiview pseudo consistency for 2D supervision, with no GT inputs."""
import numpy as np


SUPERVISION_CONTRACT = dict(version='epipolar_v1', scale_px=3.0, weight_floor=0.25,
    unknown_weight=1.0, pair_distance='symmetric_point_to_epipolar_line_rms_px',
    consensus='lower_median_over_visible_non_degenerate_peers',
    weight='floor+(1-floor)/(1+(error/scale)^2)', loss='weighted_L1_normalized_by_weight_mass',
    support='preserve_parent_query_pseudo_visibility_and_prediction_geometry_masks')


def validate_supervision(policy):
    if policy not in ('legacy', 'epipolar', 'fixed_support', 'oracle_coordinates', 'oracle_coordinates_visibility', 'retracking_aux'):
        raise ValueError('Unknown supervision_policy: ' + str(policy))


def supervision_contract(policy):
    validate_supervision(policy)
    if policy == 'retracking_aux':
        from track3d.utils.retracking_auxiliary import CONTRACT
        return CONTRACT
    if policy.startswith('oracle_'):
        from track3d.utils.oracle_supervision import CONTRACT
        return dict(CONTRACT, policy=policy)
    if policy == 'fixed_support':
        from track3d.utils.fixed_support_loss import CONTRACT
        return CONTRACT
    return SUPERVISION_CONTRACT


def epipolar_confidence(intrinsics, extrinsics, pseudo_tracks, pseudo_visibility, query_t):
    """Unbatched K/RT [S,V,...], pseudo [S,V,N,2]; recomputable from fixed input.

    Missing/degenerate cross-view evidence stays neutral. A nonzero floor keeps
    contradictory observations recoverable; no smooth-motion prior is imposed.
    """
    k, rt, uv, vis = [np.asarray(x, dtype=np.float64) for x in
                      (intrinsics, extrinsics, pseudo_tracks, pseudo_visibility)]
    if uv.ndim != 4 or uv.shape[-1] != 2:
        raise ValueError('Expected pseudo tracks [S,V,N,2]')
    s, v, n, _ = uv.shape
    q = np.asarray(query_t)
    if k.shape != (s,v,3,3) or rt.shape != (s,v,4,4) or vis.shape != (s,v,n) or q.shape != (n,):
        raise ValueError('Mismatched camera/pseudo/query dimensions')
    if not all(np.isfinite(a).all() for a in (k,rt,uv,vis,q)):
        raise ValueError('Nonfinite pseudo confidence input')
    if np.any(q != np.rint(q)) or np.any(q < 0) or np.any(q >= s):
        raise ValueError('Invalid query times')
    inverse = np.linalg.inv(k)
    hom = np.concatenate([uv,np.ones((s,v,n,1))],axis=-1)
    pair = np.full((s,v,n,v),np.inf)
    available = (vis > .5) & (np.arange(s)[:,None,None] >= q[None,None,:])
    for a in range(v):
        for b in range(a+1,v):
            rotation = rt[:,b,:3,:3] @ rt[:,a,:3,:3].transpose(0,2,1)
            translation = rt[:,b,:3,3] - np.einsum('sij,sj->si',rotation,rt[:,a,:3,3])
            skew = np.zeros((s,3,3))
            x,y,z = translation.T
            skew[:,0,1],skew[:,0,2],skew[:,1,0] = -z,y,z
            skew[:,1,2],skew[:,2,0],skew[:,2,1] = -x,-y,x
            f = inverse[:,b].transpose(0,2,1) @ skew @ rotation @ inverse[:,a]
            norm = np.linalg.norm(f,axis=(1,2))
            f = f / np.maximum(norm[:,None,None],1e-15)
            line_b = np.einsum('sij,snj->sni',f,hom[:,a])
            line_a = np.einsum('sji,snj->sni',f,hom[:,b])
            numerator = np.abs(np.sum(hom[:,b]*line_b,axis=-1))
            na = np.sum(line_a[...,:2]**2,axis=-1)
            nb = np.sum(line_b[...,:2]**2,axis=-1)
            valid = available[:,a] & available[:,b] & (norm[:,None]>1e-12) & (na>1e-12) & (nb>1e-12)
            distance = numerator * np.sqrt(.5*(1/np.maximum(na,1e-12)+1/np.maximum(nb,1e-12)))
            distance = np.where(valid,distance,np.inf)
            pair[:,a,:,b],pair[:,b,:,a] = distance,distance
    count = np.isfinite(pair).sum(-1)
    ordered = np.sort(pair,axis=-1)
    index = np.maximum(count-1,0)//2
    error = np.take_along_axis(ordered,index[...,None],axis=-1)[...,0]
    known = count > 0
    floor,scale = SUPERVISION_CONTRACT['weight_floor'],SUPERVISION_CONTRACT['scale_px']
    weight = np.where(known,floor+(1-floor)/(1+(error/scale)**2),1.)
    return dict(weights=weight.astype(np.float32), error_px=np.where(known,error,np.nan).astype(np.float32),
                peer_count=count.astype(np.int16))
