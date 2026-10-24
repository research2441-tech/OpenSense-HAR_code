"""Temporal alignment and resampling of skeleton + inertial streams.

Window = Kinect frames [t0, t0 + (F-1)/30 s]; inertial streams are bin-averaged
onto the same window (anti-aliased decimation).  Output: cache/seq.npz
"""
import pickle, numpy as np
from scipy.optimize import linear_sum_assignment

TS, TU = 64, 128
r = pickle.load(open('cache/raw.pkl', 'rb'))


def resample_lin(x, T):
    F = x.shape[0]
    src = np.linspace(0, 1, F); dst = np.linspace(0, 1, T)
    flat = x.reshape(F, -1)
    out = np.stack([np.interp(dst, src, flat[:, j]) for j in range(flat.shape[1])], 1)
    return out.reshape((T,) + x.shape[1:])


def bin_mean(t, v, t0, t1, T):
    edges = np.linspace(t0, t1, T + 1)
    idx = np.clip(np.searchsorted(edges, t, side='right') - 1, -1, T)
    out = np.full((T, v.shape[1]), np.nan, np.float32)
    for b in range(T):
        m = idx == b
        if m.any():
            out[b] = v[m].mean(0)
    # fill empty bins by interpolation over centres
    cen = 0.5 * (edges[:-1] + edges[1:])
    for j in range(v.shape[1]):
        good = ~np.isnan(out[:, j])
        if good.sum() == 0:
            out[:, j] = np.interp(cen, t, v[:, j])
        else:
            out[:, j] = np.interp(cen, cen[good], out[good, j])
    return out


S, U, Y, SUB, REP, DUR, FR = [], [], [], [], [], [], []
subs = ['cx', 'cyy', 'myj', 'qyh', 'zyh']
for x in r:
    sk = x['skel']; F = len(sk)
    t0 = (x['skel_t'][0] % 1e12) / 1e6
    t1 = t0 + (F - 1) / 30.0
    imus = []
    for m in x['imu']:
        t = m[:, 6] / 1e6
        o = np.argsort(t); t = t[o]; v = m[o, :6]
        # clip window to available IMU span
        a, b = max(t0, t[0]), min(t1, t[-1])
        if b - a < 0.5:  # misaligned clock: fall back to IMU's own span end-aligned
            a, b = t[-1] - (t1 - t0), t[-1]
        imus.append(bin_mean(t, v, a, b, TU))
    U.append(np.concatenate(imus, 1))
    S.append(resample_lin(sk, TS))
    Y.append(x['action'] - 1); SUB.append(subs.index(x['subj'])); REP.append(x['rep'])
    DUR.append(t1 - t0); FR.append(F)
S = np.array(S, np.float32); U = np.array(U, np.float32)
print('nan S', np.isnan(S).any(axis=(1,2,3)).sum(), 'nan U', np.isnan(U).any(axis=(1,2)).sum())
print(S.shape, U.shape)

# --- recover IMU placement by kinematic correlation with Kinect joints ---
# Kinect v2 joint ids: 5 ElbowL 6 WristL 9 ElbowR 10 WristR 20 SpineShoulder(chest) 1 SpineMid(abdomen)
# 13 KneeL 14 AnkleL 17 KneeR 18 AnkleR
cand = {'L wrist': 6, 'L elbow': 5, 'R wrist': 10, 'R elbow': 9, 'chest': 20, 'abdomen': 1,
        'L knee': 13, 'L ankle': 14, 'R knee': 17, 'R ankle': 18}
names = list(cand)
C = np.zeros((10, 10))
for i in range(len(S)):
    sp = np.linalg.norm(np.diff(S[i], axis=0), axis=-1)  # (T-1, 25) joint speed
    sp = resample_lin(sp, TU)
    for k in range(10):
        g = np.linalg.norm(U[i][:, 6 * k + 3:6 * k + 6], axis=-1)
        for j, nm in enumerate(names):
            a = sp[:, cand[nm]]
            if np.isfinite(a).all() and np.isfinite(g).all() and a.std() > 1e-6 and g.std() > 1e-6:
                C[k, j] += np.nan_to_num(np.corrcoef(a.astype(np.float64), g.astype(np.float64))[0, 1])
C /= len(S)
ri, ci = linear_sum_assignment(-C)
placement = {int(k): names[j] for k, j in zip(ri, ci)}
print('placement', placement)
print(np.round(C, 2))
np.savez_compressed('cache/seq.npz', S=S, U=U, Y=np.array(Y), SUB=np.array(SUB), REP=np.array(REP),
                    DUR=np.array(DUR), FR=np.array(FR), corr=C,
                    placement=np.array([placement[k] for k in range(10)]))
