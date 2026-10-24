"""Data loading, partitioning, augmentation and controlled shifts for CZU-MHAD.

Modalities: 'v' depth video (T_v=16 frames, 32x32, 2 channels: depth + motion),
            's' Kinect skeleton (T_s=64, 25 joints x 3 -> 150 features incl. velocity),
            'u' ten body-worn IMUs (T_u=128, 60 channels).
"""
import numpy as np

MODS = ('v', 's', 'u')
NCLS = 22
SUBJ = ['cx', 'cyy', 'myj', 'qyh', 'zyh']
# IMU unit -> body site, recovered from per-action activity (see prep_seq.py / paper Sec. IV-A)
IMU_SITE = ['chest', 'abdomen', 'L elbow', 'L wrist', 'R elbow', 'R wrist', 'L knee', 'L ankle', 'R knee', 'R ankle']
UPPER = [2, 3, 4, 5]          # arm units
LOWER = [6, 7, 8, 9]          # leg units
TRUNK = [0, 1]
YAW_VIEWS = (0, 30, 60)       # degrees, synthetic yaw re-projection of depth + skeleton

# ---------------------------------------------------------------- loading


def rot_y(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)


class Bank:
    """Holds all sequences; returns modality tensors for a given view."""

    def __init__(self, root='cache'):
        d = np.load(f'{root}/seq.npz')
        self.S = d['S']; self.U = d['U']; self.Y = d['Y'].astype(np.int32)
        self.SUB = d['SUB'].astype(np.int32); self.REP = d['REP']
        self.N = len(self.Y)
        try:
            v = np.load(f'{root}/depth.npz')
            self.V = {int(k[1:]): v[k] for k in v.files if k.startswith('v')}
            self.centre = v['centre']
        except FileNotFoundError:
            self.V = None
            self.centre = self.S[:, 0, 0, :].copy()
        t = np.load(f'{root}/text_tok.npz')
        self.TT, self.TM, self.LT, self.LM = t['TT'], t['TM'], t['LT'], t['LM']

    def skeleton(self, idx, yaw=0):
        S = self.S[idx]                                   # (n,64,25,3)
        c = self.centre[idx][:, None, None, :]
        X = S - c
        if yaw:
            X = X @ rot_y(yaw).T
        X = X - X[:, :1, :1, :]                           # translate to first-frame spine base
        vel = np.diff(X, axis=1, prepend=X[:, :1]) * 10.0
        return np.concatenate([X, vel], -1).reshape(len(idx), 64, 150).astype(np.float32)

    def inertial(self, idx):
        U = self.U[idx].copy()
        U[..., 3::6] /= 100.0; U[..., 4::6] /= 100.0; U[..., 5::6] /= 100.0   # gyro to hecto-deg/s
        return U.astype(np.float32)

    def visual(self, idx, yaw=0):
        if self.V is None:
            return np.zeros((len(idx), 16, 32, 32, 2), np.float32)
        D = self.V[yaw][idx][:, ::2].astype(np.float32) / 1000.0  # metres rel. to body centre, (n,16,32,32)
        mot = np.diff(D, axis=1, prepend=D[:, :1]) * 4.0
        return np.stack([D, mot], -1)

    def batch(self, idx, yaw=0):
        idx = np.asarray(idx)
        return {'v': self.visual(idx, yaw), 's': self.skeleton(idx, yaw), 'u': self.inertial(idx)}


# ---------------------------------------------------------------- partitions

CLASS_SPLITS = {
    # 'seen' trains; 'desc' unseen but described (zero-shot retrieval); 'unk' undescribed (must be rejected)
    'mirror': dict(desc=[1, 7, 13], unk=[11, 19]),        # L high wave, L fork, L kick side | L kick fwd, R turning
    'shuffle': dict(desc=[2, 9, 12], unk=[5, 16]),        # R horiz wave, L circle, R kick side | grasp R, wave up-down
    'holistic': dict(desc=[14, 15, 17], unk=[18, 21]),    # clap, bend down, sur place | L turning, R lateral
}


def class_split(name):
    cs = CLASS_SPLITS[name]
    seen = [c for c in range(NCLS) if c not in cs['desc'] + cs['unk']]
    return np.array(seen), np.array(cs['desc']), np.array(cs['unk'])


def loso(bank, test_subj, seed, val_frac=0.15, classes=None):
    """Leave-one-subject-out split. Validation = held-out repetitions of training subjects."""
    rng = np.random.RandomState(1000 + seed)
    tr_all = np.where(bank.SUB != test_subj)[0]
    te = np.where(bank.SUB == test_subj)[0]
    if classes is not None:
        tr_all = tr_all[np.isin(bank.Y[tr_all], classes)]
    # stratified by (subject, class): hold out a fraction of repetitions
    val = []
    for s in np.unique(bank.SUB[tr_all]):
        for c in np.unique(bank.Y[tr_all]):
            ids = tr_all[(bank.SUB[tr_all] == s) & (bank.Y[tr_all] == c)]
            if len(ids) > 3:
                k = max(1, int(round(val_frac * len(ids))))
                val.extend(rng.choice(ids, k, replace=False))
    val = np.array(sorted(val))
    tr = np.setdiff1d(tr_all, val)
    return tr, val, te


# ---------------------------------------------------------------- augmentation (training)


def augment(x, rng, p_corrupt=0.3):
    """Mild geometric/sensor augmentation plus random stream corruption, applied identically for
    every fusion model so that robustness gains are not an artefact of augmentation."""
    n = x['s'].shape[0]
    out = {k: v.copy() for k, v in x.items()}
    # amplitude jitter
    out['u'] *= rng.uniform(0.9, 1.1, (n, 1, out['u'].shape[2])).astype(np.float32)
    out['s'][..., :75] *= rng.uniform(0.95, 1.05, (n, 1, 1)).astype(np.float32)
    for i in range(n):
        if rng.rand() < p_corrupt:
            m = MODS[rng.randint(3)]
            sev = rng.randint(1, 4)
            out[m][i] = corrupt_one(out[m][i], m, rng.choice(['noise', 'occlude']), sev, rng)
    return out


# ---------------------------------------------------------------- corruptions (test-time shifts)


def corrupt_one(a, m, kind, sev, rng):
    a = a.copy()
    s = sev / 5.0
    if m == 'u':
        if kind == 'noise':
            a += rng.normal(0, 0.15 * sev, a.shape).astype(np.float32)
        elif kind == 'occlude':   # dead units (flat-lined)
            k = max(1, int(round(2 * sev)))
            for j in rng.choice(10, min(k, 10), replace=False):
                a[:, 6 * j:6 * j + 6] = 0.0
        elif kind == 'drift':     # bias + gain drift
            a = a * (1 + 0.15 * sev) + np.linspace(0, 0.2 * sev, a.shape[0])[:, None].astype(np.float32)
        elif kind == 'clip':
            a = np.clip(a, -2.0 / sev, 2.0 / sev)
    elif m == 's':
        if kind == 'noise':
            a[:, :75] += rng.normal(0, 0.02 * sev, (a.shape[0], 75)).astype(np.float32)
            a[:, 75:] = np.diff(a[:, :75], axis=0, prepend=a[:1, :75]) * 10.0
        elif kind == 'occlude':   # limb tracking loss: zero a random limb chain
            chains = [[4, 5, 6, 7, 21, 22], [8, 9, 10, 11, 23, 24], [12, 13, 14, 15], [16, 17, 18, 19]]
            for c in rng.choice(4, min(sev, 4), replace=False):
                for j in chains[c]:
                    a[:, 3 * j:3 * j + 3] = 0.0; a[:, 75 + 3 * j:75 + 3 * j + 3] = 0.0
        elif kind == 'drop':      # frame dropping with hold
            keep = rng.rand(a.shape[0]) > 0.15 * sev
            keep[0] = True
            last = 0
            for t in range(a.shape[0]):
                if keep[t]: last = t
                else: a[t] = a[last]
    elif m == 'v':
        if kind == 'noise':
            a[..., 0] += rng.normal(0, 0.05 * sev, a[..., 0].shape).astype(np.float32)
            a[..., 1] = np.diff(a[..., 0], axis=0, prepend=a[:1, ..., 0]) * 4.0
        elif kind == 'occlude':   # static occluding patch
            h = int(6 + 4 * sev)
            r0, c0 = rng.randint(0, 32 - h + 1, 2)
            a[:, r0:r0 + h, c0:c0 + h, :] = 0.0
        elif kind == 'holes':
            msk = rng.rand(*a.shape[:3]) < 0.1 * sev
            a[msk] = 0.0
    return a


def corrupt_batch(x, m, kind, sev, seed):
    rng = np.random.RandomState(seed)
    out = {k: v.copy() for k, v in x.items()}
    for i in range(out[m].shape[0]):
        out[m][i] = corrupt_one(out[m][i], m, kind, sev, rng)
    return out


def mask_dict(n, avail):
    """avail: iterable of available modalities -> per-sample availability mask (n,3)."""
    return np.tile(np.array([[1.0 if m in avail else 0.0 for m in MODS]], np.float32), (n, 1))
