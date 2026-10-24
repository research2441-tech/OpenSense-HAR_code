"""Depth-video stream: person segmentation guided by the Kinect skeleton, back-projection to a point
cloud, optional yaw re-projection, and orthographic rendering to 32x32 body-centred depth maps.

Output cache/depth.npz: v0, v30, v60 (N,32,32,32) int16 [mm, relative to body centre; 1000 = empty],
centre (N,3) body centre in camera space (spine-mid, first frame).
"""
import os, glob, subprocess, pickle, numpy as np
from PIL import Image

ROOT = 'data/depth'
FX, CX, CY = 365.0, 256.0, 212.0
TV, G = 32, 32
YAWS = (0, 30, 60)
recs = pickle.load(open('cache/raw.pkl', 'rb'))
seq = np.load('cache/seq.npz')


def rot_y(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)


def scale_estimate(sample):
    """Estimate metres-per-grey-level from joint depths vs. image values (median ratio)."""
    r = []
    for x, fs, ft in sample:
        st = x['skel_t']
        for i in range(0, len(fs), 15):
            j = np.argmin(abs(st - ft[i]))
            J = x['skel'][j]
            a = np.array(Image.open(fs[i]))
            u = np.clip((CX + FX * J[:, 0] / J[:, 2]).astype(int), 0, 511)
            v = np.clip((CY - FX * J[:, 1] / J[:, 2]).astype(int), 0, 423)
            d = a[v, u].astype(float)
            ok = (d > 20) & (abs(d - np.median(d)) < 6)
            r.extend(list(J[ok, 2] / d[ok]))
    return float(np.median(r))


def render(P, yaw):
    if yaw:
        P = P @ rot_y(yaw).T
    gx = np.floor((P[:, 0] + 1.0) / 2.0 * G).astype(int)
    gy = np.floor((0.9 - P[:, 1]) / 2.0 * G).astype(int)
    ok = (gx >= 0) & (gx < G) & (gy >= 0) & (gy < G)
    img = np.full(G * G, 1000.0, np.float32)
    np.minimum.at(img, gy[ok] * G + gx[ok], P[ok, 2] * 1000.0)
    return img.reshape(G, G)


def process(x, fs, ft, k):
    sk = x['skel']; st = x['skel_t']
    F = len(sk)
    tgt = np.linspace(st[0], st[0] + (F - 1) / 30.0 * 1e6, TV)
    c = sk[0, 1].copy()                                  # spine-mid, first frame
    out = {y: np.zeros((TV, G, G), np.int16) for y in YAWS}
    vv, uu = np.mgrid[0:424, 0:512]
    for t, tt in enumerate(tgt):
        i = int(np.argmin(abs(ft - tt)))
        j = int(np.argmin(abs(st - ft[i])))
        J = sk[j]
        a = np.array(Image.open(fs[i])).astype(np.float32)
        Z = a * k
        zc = J[1, 2]
        m = (a > 10) & (abs(Z - zc) < 0.5)
        X = (uu - CX) * Z / FX; Y = -(vv - CY) * Z / FX
        m &= (X > J[:, 0].min() - 0.35) & (X < J[:, 0].max() + 0.35) & (Y > J[:, 1].min() - 0.25) & (Y < J[:, 1].max() + 0.3)
        P = np.stack([X[m], Y[m], Z[m]], 1) - c
        for y in YAWS:
            out[y][t] = np.round(render(P, y)).astype(np.int16)
    return out, c


def main():
    N = len(recs)
    V = {y: np.zeros((N, TV, G, G), np.int16) for y in YAWS}
    C = np.zeros((N, 3), np.float32)
    k = None
    for s in ['cx', 'cyy', 'myj', 'qyh', 'zyh']:
        subprocess.run(['git', '-C', ROOT, 'sparse-checkout', 'set', f'depth/{s}'], check=True)
        idx = [i for i, r in enumerate(recs) if r['subj'] == s]
        lst = []
        for i in idx:
            x = recs[i]
            fs = sorted(glob.glob(f"{ROOT}/depth/{s}/action_{x['action']}/time_{x['rep']}/*.png"))
            ft = np.array([int(os.path.basename(f)[:-4]) for f in fs], np.float64)
            lst.append((i, x, fs, ft))
        if k is None:
            k = scale_estimate([(x, fs, ft) for _, x, fs, ft in lst[:40] if len(fs)])
            print('metres per level', k, flush=True)
        miss = 0
        for i, x, fs, ft in lst:
            if len(fs) < 5:
                miss += 1
                V[0][i] = V[30][i] = V[60][i] = 1000
                C[i] = x['skel'][0, 1]
                continue
            o, c = process(x, fs, ft, k)
            for y in YAWS:
                V[y][i] = o[y]
            C[i] = c
        print(s, len(lst), 'missing depth', miss, flush=True)
    np.savez_compressed('cache/depth.npz', v0=V[0], v30=V[30], v60=V[60], centre=C, k=k)


if __name__ == '__main__':
    main()
