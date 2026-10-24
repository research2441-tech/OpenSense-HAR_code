"""Parse CZU-MHAD skeleton and inertial text files into one pickle.

Skeleton file: one line per Kinect frame, 25 joints x (x, y, z, t).
Inertial file (pi00..pi09): one line per sample, ax ay az [g], gx gy gz [deg/s], t.
"""
import os, sys, pickle, numpy as np

ROOT_S = 'data/skel/skeleton'
ROOT_I = 'data/sens/sensors'
SUBJ = ['cx', 'cyy', 'myj', 'qyh', 'zyh']


def read_txt(p):
    rows = []
    with open(p, 'r', errors='ignore') as f:
        for ln in f:
            v = ln.split()
            if v:
                try:
                    rows.append([float(a) for a in v])
                except ValueError:
                    pass
    if not rows:
        return None
    n = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    return np.array([r for r in rows if len(r) == n], dtype=np.float64)


def main(out):
    recs = []
    for s in SUBJ:
        for a in range(1, 23):
            d = os.path.join(ROOT_S, s, f'action_{a}')
            if not os.path.isdir(d):
                continue
            for tdir in sorted(os.listdir(d), key=lambda x: int(x.split('_')[1])):
                sk = read_txt(os.path.join(d, tdir))
                if sk is None or sk.shape[1] != 100:
                    print('bad skel', s, a, tdir); continue
                sk = sk.reshape(-1, 25, 4)
                imu = []
                ok = True
                for k in range(10):
                    p = os.path.join(ROOT_I, s, f'action_{a}', tdir, f'pi{k:02d}')
                    if not os.path.exists(p):
                        ok = False; break
                    m = read_txt(p)
                    if m is None or m.shape[1] != 7 or len(m) < 20:
                        ok = False; break
                    imu.append(m.astype(np.float32))
                if not ok:
                    print('bad imu', s, a, tdir); continue
                recs.append(dict(subj=s, action=a, rep=int(tdir.split('_')[1]),
                                 skel=sk[:, :, :3].astype(np.float32), skel_t=sk[:, 0, 3], imu=imu))
    print('records', len(recs))
    pickle.dump(recs, open(out, 'wb'))


if __name__ == '__main__':
    main(sys.argv[1])
