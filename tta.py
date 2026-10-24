"""Sequential cross-domain deployment with continual test-time adaptation.

Source model: OpenSense-HAR trained on four subjects (closed set, seed 0).  The held-out subject is
streamed through six domains; each batch is predicted first and only then used for adaptation.
Methods: source (frozen), Tent, EATA, CoTTA, OpenSense-TTA (adapters + prototype memory), and two
ablations of the latter.  python3 tta.py <fold>
"""
import sys, os, pickle, time
import numpy as np, jax, jax.numpy as jnp, optax
import data as Dt, train as TR, models as M
import run as RN
from flax.traverse_util import flatten_dict, unflatten_dict

BSZ = 16
DOMAINS = [('D0 clean', 0, []), ('D1 view 30', 30, []), ('D2 view 30 + IMU drift', 30, [('u', 'drift', 3)]),
           ('D3 view 60 + skeleton noise', 60, [('s', 'noise', 3)]),
           ('D4 view 60 + depth holes + dead IMUs', 60, [('v', 'holes', 3), ('u', 'occlude', 2)]),
           ('D5 clean (revisit)', 0, [])]


def entropy(p):
    return -(p * jnp.log(p + 1e-8)).sum(-1)


def main(fold, mode='closed'):
    bank = Dt.Bank()
    if mode == 'closed':
        seen = np.arange(22)
        tr_idx, va_idx, te_idx = Dt.loso(bank, fold, 0)
        P = pickle.load(open(f'results/closed/params_opensense_f{fold}_s0.pkl', 'rb'))
        R = pickle.load(open(f'results/closed/opensense_f{fold}_s0.pkl', 'rb'))
        vl = R['val']['logits']
    else:
        seen, desc, unk = Dt.class_split('mirror')
        tr_idx, va_idx, _ = Dt.loso(bank, fold, 0, classes=seen)
        te_idx = np.where(bank.SUB == fold)[0]
        P = pickle.load(open(f'results/open/params_opensense_f{fold}_s0_mirror.pkl', 'rb'))
        R = pickle.load(open(f'results/open/opensense_f{fold}_s0_mirror.pkl', 'rb'))
        vl = R['out_va']['logits']
        vl = vl if vl.shape[1] == len(seen) else vl[:, seen]
    cfg = {k: v for k, v in RN.CONFIGS['opensense'].items() if k != 'kind'}
    trn = TR.OSTrainer(cfg, bank, seen, 0)
    trn.norm = TR.Norm(bank, tr_idx)
    params0 = P['params']
    # adapters are unused during training; initialise them here (LoRA-style: A ~ N(0, 1/D), B = 0)
    fl = flatten_dict(params0)
    kk = jax.random.PRNGKey(99 + fold)
    for i, k in enumerate(sorted(fl)):
        if k[0].startswith('adapt') and k[1] == 'A':
            fl[k] = jax.random.normal(jax.random.fold_in(kk, i), fl[k].shape) / np.sqrt(fl[k].shape[0])
        if k[0].startswith('adapt') and k[1] == 'B':
            fl[k] = jnp.zeros_like(fl[k])
    params0 = unflatten_dict(fl)
    T = jnp.asarray(bank.TT[seen]); Mk = jnp.asarray(bank.TM[seen])
    C = len(seen)
    cmap = -np.ones(22, int); cmap[seen] = np.arange(C)
    model = trn.model
    E_val = -np.log(np.exp(vl - vl.max(1, keepdims=True)).sum(1)) - vl.max(1)
    delta_E = float(np.quantile(E_val, 0.95))
    h0 = 0.4 * np.log(C)
    # source prototype memory from training embeddings (clean)
    Xtr = trn.norm(bank.batch(tr_idx))
    otr = trn.predict(params0, Xtr, mc=1)
    mu0 = np.stack([otr['ef'][bank.Y[tr_idx] == c].mean(0) for c in seen])
    mu0 = mu0 / np.linalg.norm(mu0, axis=1, keepdims=True)
    tproto = np.asarray(otr['P'])
    # domain streams
    rng = np.random.RandomState(fold)
    streams = []
    for name, yaw, cors in DOMAINS:
        x = trn.norm(bank.batch(te_idx, yaw))
        for (m, k, s) in cors:
            x = Dt.corrupt_batch(x, m, k, s, seed=50 + fold)
        order = rng.permutation(len(te_idx))
        streams.append((name, {k: v[order] for k, v in x.items()}, cmap[bank.Y[te_idx][order]]))

    flat0 = flatten_dict(params0)
    is_ln = {k: any('LayerNorm' in p for p in k) for k in flat0}
    is_ad = {k: any(p.startswith('adapt') for p in k) for k in flat0}

    def fwd(params, x, adapt):
        o = model.apply({'params': params}, x, jnp.ones((x['s'].shape[0], 3)), T, Mk, train=False, adapt=adapt,
                        rngs={'dropout': jax.random.PRNGKey(0)})
        return o['logits'], o['ef'], o['tau']

    def masked_update(grads, which):
        g = flatten_dict(grads)
        return unflatten_dict({k: (v if which[k] else jnp.zeros_like(v)) for k, v in g.items()})

    # Fisher information of LN parameters for EATA (pseudo-labelled clean validation data)
    Xva = trn.norm(bank.batch(va_idx[:64]))

    def ce_pl(params, x):
        lg, _, _ = fwd(params, x, False)
        yp = jax.lax.stop_gradient(lg.argmax(1))
        return optax.softmax_cross_entropy_with_integer_labels(lg, yp).mean()
    gF = flatten_dict(jax.grad(ce_pl)(params0, TR.to_j(Xva)))
    fisher = {k: (v ** 2) for k, v in gF.items()}

    results = {}
    methods = ['source', 'tent', 'eata', 'cotta', 'ours', 'ours_nosel', 'ours_nomem', 'ours_noad'] if mode == 'closed' \
        else ['source', 'tent', 'eata', 'ours', 'ours_nosel']
    for method in methods:
        params = params0
        teacher = params0
        adapt = method.startswith('ours')
        which = is_ad if adapt else is_ln
        opt = optax.adam(1e-4 if not adapt else 1e-3)
        st = opt.init(params)
        mu = mu0.copy()

        def loss_fn(params, x, sel_w, protos, method=method, teacher=teacher):
            lg, ef, tau = fwd(params, x, adapt)
            p = jax.nn.softmax(lg, -1)
            H = entropy(p)
            if method == 'tent':
                return H.mean()
            if method == 'eata':
                fl = flatten_dict(params)
                reg = sum((fisher[k] * (fl[k] - flat0[k]) ** 2).sum() for k in fl if is_ln[k])
                return (sel_w * H).sum() / (sel_w.sum() + 1e-6) + 2000.0 * reg
            if method == 'cotta':
                return -(sel_w * jnp.log(p + 1e-8)).sum(-1).mean()        # sel_w = teacher soft labels
            # ours: confident subset, entropy + prototype anchoring + L2 anchor on adapter weights
            yh = jax.lax.stop_gradient(lg.argmax(1))
            anc = 1.0 - (ef * protos[yh]).sum(-1)
            fl = flatten_dict(params)
            reg = sum(((fl[k] - flat0[k]) ** 2).sum() for k in fl if is_ad[k])
            return (sel_w * (H + 1.0 * anc)).sum() / (sel_w.sum() + 1e-6) + 1.0 * reg

        grad_fn = jax.jit(jax.grad(loss_fn))
        fwd_j = jax.jit(fwd, static_argnums=(2,))
        accs = []; unk_used = []; n_sel = []
        for name, x, y in streams:
            correct = 0; known = 0; uu = 0; ns = 0
            for b in range(0, len(y), BSZ):
                xb = {k: v[b:b + BSZ] for k, v in x.items()}
                if len(xb['s']) < BSZ:
                    pad = BSZ - len(xb['s'])
                    xb = {k: np.concatenate([v, np.repeat(v[-1:], pad, 0)]) for k, v in xb.items()}
                nb = min(BSZ, len(y) - b)
                xj = TR.to_j(xb)
                lg, ef, tau = fwd_j(params, xj, adapt)
                lg = np.asarray(lg); ef = np.asarray(ef)
                if method in ('ours', 'ours_nosel', 'ours_noad'):
                    protos = np.asarray(TR.M.l2n(jnp.asarray(0.5 * tproto + 0.5 * mu)))
                    lgp = ef @ protos.T / float(tau)
                else:
                    protos = tproto
                    lgp = lg
                if method == 'cotta':
                    tl_, _, _ = fwd_j(teacher, xj, False)
                    pred = np.asarray(tl_).argmax(1)
                else:
                    pred = lgp.argmax(1)
                yb = y[b:b + nb]
                correct += int(((pred[:nb] == yb) & (yb >= 0)).sum()); known += int((yb >= 0).sum())
                if method == 'source':
                    continue
                prob = np.exp(lgp - lgp.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
                Hb = -(prob * np.log(prob + 1e-8)).sum(1)
                Eb = -np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) - lg.max(1)
                valid = np.zeros(BSZ); valid[:nb] = 1
                if method == 'eata':
                    sel = valid * (Hb < h0) * np.exp(-(Hb - h0))
                elif method == 'cotta':   # augmentation-averaged teacher pseudo-labels
                    arng = np.random.RandomState(b + 13 * len(accs))
                    tp = 0.0
                    for _ in range(4):
                        xa = Dt.augment({k: np.asarray(v) for k, v in xb.items()}, arng, p_corrupt=0.0)
                        xa['u'] = xa['u'] + arng.normal(0, 0.05, xa['u'].shape).astype(np.float32)
                        tl, _, _ = fwd_j(teacher, TR.to_j(xa), False)
                        tp = tp + np.asarray(jax.nn.softmax(tl, -1)) / 4
                    sel = tp * valid[:, None]
                elif method == 'ours_nosel':
                    sel = valid
                elif method.startswith('ours'):
                    sel = valid * (Eb <= delta_E) * (Hb <= h0)
                else:
                    sel = valid
                selm = sel if sel.ndim == 1 else valid
                yfull = np.concatenate([yb, -np.ones(BSZ - nb, int)])
                uu += int(((selm > 0) & (yfull < 0) & (valid > 0)).sum()); ns += int((selm > 0).sum())
                if method == 'ours_noad':
                    g = None
                else:
                    g = grad_fn(params, xj, jnp.asarray(sel), jnp.asarray(protos))
                if g is None:
                    pass
                else:
                    g = masked_update(g, which)
                    up, st = opt.update(g, st, params)
                    params = optax.apply_updates(params, up)
                if False:
                    pass
                if method == 'cotta':   # EMA teacher + stochastic restoration of LN parameters
                    teacher = jax.tree_util.tree_map(lambda t_, s_: 0.99 * t_ + 0.01 * s_, teacher, params)
                    fp = flatten_dict(params)
                    kk = jax.random.PRNGKey(b + 7 * len(accs))
                    fp = {k: (jnp.where(jax.random.bernoulli(jax.random.fold_in(kk, i), 0.01, v.shape), flat0[k], v)
                              if is_ln[k] else v) for i, (k, v) in enumerate(fp.items())}
                    params = unflatten_dict(fp)
                if method in ('ours', 'ours_nosel', 'ours_noad') and sel.sum() > 0:
                    # prototype memory update from confident samples with bounded drift
                    for i in np.where(sel > 0)[0]:
                        c = pred[i]
                        m_new = 0.95 * mu[c] + 0.05 * ef[i]
                        m_new /= np.linalg.norm(m_new)
                        if m_new @ mu0[c] >= 0.8:
                            mu[c] = m_new
            accs.append(correct / max(known, 1)); unk_used.append(uu); n_sel.append(ns)
        results[method] = accs
        results[method + '_unk_used'] = unk_used
        results[method + '_n_sel'] = n_sel
        print(fold, mode, method, ' '.join(f'{a:.3f}' for a in accs), 'unk_used', unk_used, flush=True)
    os.makedirs('results/tta', exist_ok=True)
    pickle.dump({'results': results, 'domains': [d[0] for d in DOMAINS], 'delta_E': delta_E},
                open(f'results/tta/tta_{mode}_f{fold}.pkl', 'wb'))


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else 'closed')
