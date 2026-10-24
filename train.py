"""Training / inference utilities for OpenSense-HAR and baselines."""
import time, functools
import numpy as np, jax, jax.numpy as jnp, optax
import flax.linen as nn
from flax.core import unfreeze
import models as M
import data as Dt

BS = 32


def to_j(x):
    return {k: jnp.asarray(v) for k, v in x.items()}


# ------------------------------------------------------------------ normalisation stats (train only)

class Norm:
    def __init__(self, bank, idx):
        x = bank.batch(idx)
        self.mu = {k: x[k].reshape(-1, x[k].shape[-1]).mean(0) for k in ('s', 'u')}
        self.sd = {k: x[k].reshape(-1, x[k].shape[-1]).std(0) + 1e-3 for k in ('s', 'u')}

    def __call__(self, x):
        y = dict(x)
        for k in ('s', 'u'):
            y[k] = ((x[k] - self.mu[k]) / self.sd[k]).astype(np.float32)
        return y


def drop_mask(n, rng, p=0.15):
    a = (rng.rand(n, 3) > p).astype(np.float32)
    z = a.sum(1) == 0
    a[z, rng.randint(0, 3, z.sum())] = 1.0
    return a


# ------------------------------------------------------------------ OpenSense / OSHAR-family trainer

class OSTrainer:
    def __init__(self, cfg, bank, seen, seed, text='desc'):
        self.cfg = cfg
        self.bank = bank
        self.seen = np.asarray(seen)
        self.seed = seed
        self.model = M.OSHAR(lang=cfg.get('lang', True), rel=cfg.get('rel', True), gen=cfg.get('gen', 'latent'),
                             hier=cfg.get('hier', True), experts=cfg.get('experts', 'vsuj'), ncls=len(seen),
                             fusion=cfg.get('fusion', 'rba'), resid=cfg.get('resid', False),
                             dual=cfg.get('dual', False))
        if text == 'label':
            self.T, self.Mk = bank.LT, bank.LM
        else:
            self.T, self.Mk = bank.TT, bank.TM
        self.text_mode = text

    # ---------------------------------------------------------- init
    def init(self, xb):
        key = jax.random.PRNGKey(self.seed)
        n = xb['s'].shape[0]
        av = jnp.ones((n, 3))
        T = jnp.asarray(self.T[self.seen]); Mk = jnp.asarray(self.Mk[self.seen])
        params = self.model.init({'params': key, 'dropout': key}, to_j(xb), av, T, Mk,
                                 method=M.OSHAR.init_all)['params']
        return params

    # ---------------------------------------------------------- losses
    def stage2_loss(self, params, x, a, a_full, y, T, Mk, rng):
        cfg = self.cfg
        out = self.model.apply({'params': params}, x, a, T, Mk, train=True, rngs={'dropout': rng})
        lg = out['logits']
        loss = optax.softmax_cross_entropy_with_integer_labels(lg, y).mean()
        keys = list(out['emb'].keys())
        aux = 0.0
        for k in keys:
            mk = out['mk'][k]
            ce = optax.softmax_cross_entropy_with_integer_labels(out['logits_m'][k], y)
            aux += (ce * mk).sum() / (mk.sum() + 1e-6)
        loss += cfg.get('w_sl', 0.5) * aux / len(keys)
        if 'logits_sem' in out:      # sensor-to-text alignment of the semantic head
            loss += cfg.get('w_sem', 1.0) * optax.softmax_cross_entropy_with_integer_labels(out['logits_sem'], y).mean()
        if True:
            P = out.get('P_text', out['P'])
            emb_c = out['es'] if 'es' in out else out['ef']
            # text -> sample direction of the sensor-language contrast
            sim = (P @ emb_c.T) / out['tau']                         # (C,B)
            pos = (y[None, :] == jnp.arange(P.shape[0])[:, None]).astype(jnp.float32)
            has = pos.sum(1) > 0
            lsm = jax.nn.log_softmax(sim, axis=1)
            t2s = -(lsm * pos).sum(1) / (pos.sum(1) + 1e-6)
            loss += cfg.get('w_sl', 0.5) * (t2s * has).sum() / (has.sum() + 1e-6)
        if self.model.rel:
            P = out['P']
            Ly = P[y]
            r = 0.0
            for k in keys:
                lv = out['lv_obs'][k]
                d2 = ((jax.lax.stop_gradient(out['emb'][k]) - Ly) ** 2).sum(-1)
                mk = out['mk'][k] if k == 'j' else a[:, 'vsu'.index(k)]
                r += ((0.5 * d2 * jnp.exp(-lv) + 0.5 * lv) * mk).sum() / (mk.sum() + 1e-6)
            loss += cfg.get('w_rel', 0.2) * r / len(keys)
        gen = self.model.gen
        if gen in ('latent', 'feat'):
            g = 0.0
            for i, m in enumerate(Dt.MODS):
                tgt = jax.lax.stop_gradient(out['z'][m])
                hid = a_full[:, i] * (1 - a[:, i])          # modality present in data but hidden
                trn = a_full[:, i]                          # always learn to predict clean latents
                mu, lv = out['gen_mu'][m], out['gen_lv'][m]
                if gen == 'latent':
                    nll = 0.5 * (((tgt - mu) ** 2) * jnp.exp(-lv) + lv).mean(-1)
                else:
                    nll = ((tgt - mu) ** 2).mean(-1)
                g += (nll * trn).sum() / (trn.sum() + 1e-6)
            loss += cfg.get('w_gen', 0.5) * g / 3
            if 'logits_gen' in out:     # semantic consistency of generated latents
                sg = 0.0
                for i, m in enumerate(Dt.MODS):
                    ce = optax.softmax_cross_entropy_with_integer_labels(out['logits_gen'][m], y)
                    sg += (ce * a_full[:, i]).sum() / (a_full[:, i].sum() + 1e-6)
                loss += cfg.get('w_gen', 0.5) * sg / 3
        if gen == 'raw':
            g = 0.0
            for i, m in enumerate(Dt.MODS):
                tgt = jax.lax.stop_gradient(self.model.apply({'params': params}, x, method=M.OSHAR.tokens)[m])
                g += ((out['raw_tok'][m] - tgt) ** 2).mean()
            loss += cfg.get('w_gen', 0.5) * g / 3
        return loss, out

    def mask_loss(self, params, x, a, pm, tgt, rng):
        out = self.model.apply({'params': params}, x, a, None, None, train=True, pmask=pm, rngs={'dropout': rng},
                               method=self._pre_forward)
        l = 0.0
        for i, m in enumerate(Dt.MODS):
            w = pm[:, i]                                     # (B,8)
            err = ((out[m] - tgt[m]) ** 2).mean(-1)
            l += (err * w).sum() / (w.sum() + 1e-6)
        return l / 3

    @staticmethod
    def _pre_forward(mdl, x, a, T, Mk, train=False, pmask=None):
        Hs, zs = mdl.encode(x, a, train=train, pmask=pmask)
        Hj, _ = mdl.joint_expert(Hs, a, train=train)
        return mdl.decode(Hj)

    # ---------------------------------------------------------- fit
    def fit(self, tr, va, epochs=None, pre_epochs=None, log=False, label_frac=1.0, unlabeled=None):
        cfg = self.cfg
        bank = self.bank
        rng = np.random.RandomState(self.seed)
        self.norm = Norm(bank, tr)
        cls_map = -np.ones(Dt.NCLS, np.int32); cls_map[self.seen] = np.arange(len(self.seen))
        self.cls_map = cls_map
        X_all = self.norm(bank.batch(np.arange(bank.N)))     # cached normalised arrays (view 0)
        self.X0 = X_all
        trl = tr
        if label_frac < 1.0:
            trl = []
            for c in self.seen:
                ids = tr[bank.Y[tr] == c]
                k = max(1, int(round(label_frac * len(ids))))
                trl.extend(rng.choice(ids, k, replace=False))
            trl = np.array(trl)
        params = self.init({k: v[:2] for k, v in X_all.items()})
        epochs = epochs or cfg.get('epochs', 60)
        pre_epochs = cfg.get('pre_epochs', 30) if pre_epochs is None else pre_epochs
        T_all = self.T[self.seen]; M_all = self.Mk[self.seen]
        # ------------- stage 1: multimodal masked modelling (labels unused)
        if cfg.get('mm_pretrain', True) and pre_epochs > 0 and 'j' in self.model.experts:
            pre_idx = tr if unlabeled is None else unlabeled
            steps = pre_epochs * max(1, len(pre_idx) // BS)
            opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, 1e-3, max(1, min(100, steps // 10)), steps), weight_decay=1e-4)
            st = opt.init(params)
            tmean = None

            @jax.jit
            def pstep(params, st, x, a, pm, tgt, key):
                l, g = jax.value_and_grad(self.mask_loss)(params, x, a, pm, tgt, key)
                up, st = opt.update(g, st, params)
                return optax.apply_updates(params, up), st, l
            key = jax.random.PRNGKey(self.seed + 7)
            for ep in range(pre_epochs):
                perm = rng.permutation(pre_idx)
                for b in range(len(perm) // BS):
                    ids = perm[b * BS:(b + 1) * BS]
                    xb = {k: v[ids] for k, v in X_all.items()}
                    xb = Dt.augment(xb, rng, p_corrupt=0.0)
                    pm = (rng.rand(BS, 3, 8) < 0.5).astype(np.float32)
                    full = rng.rand(BS, 3) < 0.2                     # whole-modality masking
                    pm[full] = 1.0
                    tg = {k: np.asarray(v) for k, v in M.raw_targets(xb).items()}
                    if tmean is None:
                        tmean = {k: (v.mean((0, 1)), v.std((0, 1)) + 1e-3) for k, v in tg.items()}
                        self.tstats = tmean
                    tg = {k: (v - tmean[k][0]) / tmean[k][1] for k, v in tg.items()}
                    key, sk = jax.random.split(key)
                    params, st, l = pstep(params, st, to_j(xb), jnp.ones((BS, 3)), jnp.asarray(pm), to_j(tg), sk)
                if log:
                    print(f'  pre ep{ep} mask-loss {float(l):.3f}', flush=True)
        # ------------- stage 2: language-grounded supervised alignment
        steps = epochs * max(1, len(trl) // BS)
        lr = cfg.get('lr', 1e-3)
        opt = optax.chain(optax.clip_by_global_norm(1.0),
                          optax.adamw(optax.warmup_cosine_decay_schedule(0, lr, max(1, min(100, steps // 10)), steps), weight_decay=1e-4))
        st = opt.init(params)
        J = self.T.shape[1]

        @jax.jit
        def step(params, st, x, a, af, y, T, Mk, key):
            (l, out), g = jax.value_and_grad(self.stage2_loss, has_aux=True)(params, x, a, af, y, T, Mk, key)
            up, st = opt.update(g, st, params)
            return optax.apply_updates(params, up), st, l
        key = jax.random.PRNGKey(self.seed + 11)
        best, best_p, bad = -1, params, 0
        p_drop = cfg.get('p_drop', float(__import__('os').environ.get('P_DROP', 0.25)))
        p_cor = cfg.get('p_corrupt', 0.3)
        for ep in range(epochs):
            perm = rng.permutation(trl)
            for b in range(max(1, len(perm) // BS)):
                ids = perm[b * BS:(b + 1) * BS]
                if len(ids) < BS:
                    ids = np.concatenate([ids, rng.choice(trl, BS - len(ids))])
                xb = {k: v[ids] for k, v in X_all.items()}
                xb = Dt.augment(xb, rng, p_corrupt=p_cor)
                a = drop_mask(BS, rng, p_drop) if p_drop > 0 else np.ones((BS, 3), np.float32)
                y = cls_map[bank.Y[ids]]
                if J > 1:
                    jj = rng.choice(J, 2, replace=False)
                    T, Mk = T_all[:, jj], M_all[:, jj]
                else:
                    T, Mk = T_all, M_all
                key, sk = jax.random.split(key)
                params, st, l = step(params, st, to_j(xb), jnp.asarray(a), jnp.ones((BS, 3)), jnp.asarray(y),
                                     jnp.asarray(T), jnp.asarray(Mk), sk)
            if ep >= epochs // 3 and (ep % 3 == 2 or ep == epochs - 1):
                acc = self.accuracy(params, va)
                if acc >= best:
                    best, best_p = acc, params
                if log:
                    print(f'  ep{ep} loss {float(l):.3f} val {acc:.3f}', flush=True)
        self.params = best_p
        self.val_acc = best
        return best_p

    # ---------------------------------------------------------- inference
    def _fwd(self, params, x, a, T, Mk, adapt, mc, key):
        return self.model.apply({'params': params}, x, a, T, Mk, train=False, adapt=adapt, mc=mc,
                                rngs={'dropout': key})

    @functools.partial(jax.jit, static_argnums=(0, 6, 7))
    def fwd(self, params, x, a, T, Mk, adapt=False, mc=1, key=None):
        o = self._fwd(params, x, a, T, Mk, adapt, mc, key)
        keep = ['logits', 'ef', 'lv_f', 'w', 'tau']
        r = {k: o[k] for k in keep}
        r['emb'] = o['emb']; r['lv'] = o['lv']; r['z'] = o['z']
        if 'P' in o:
            r['P'] = o['P']
        for k in ('es', 'P_text', 'logits_sem'):
            if k in o:
                r[k] = o[k]
        return r

    def predict(self, params, x, a=None, classes=None, adapt=False, mc=8, bs=64, seed=0):
        """x: normalised dict.  classes: class ids whose text prototypes are scored (default seen)."""
        n = x['s'].shape[0]
        if a is None:
            a = np.ones((n, 3), np.float32)
        cls = self.seen if classes is None else np.asarray(classes)
        T = jnp.asarray(self.T[cls]); Mk = jnp.asarray(self.Mk[cls])
        outs = []
        key = jax.random.PRNGKey(1234 + seed)
        for b in range(0, n, bs):
            sl = slice(b, min(n, b + bs))
            m = sl.stop - sl.start
            xb = {k: v[sl] for k, v in x.items()}
            ab = a[sl]
            if m < bs:   # pad to fixed shape
                xb = {k: np.concatenate([v, np.repeat(v[-1:], bs - m, 0)]) for k, v in xb.items()}
                ab = np.concatenate([ab, np.repeat(ab[-1:], bs - m, 0)])
            key, sk = jax.random.split(key)
            o = self.fwd(params, to_j(xb), jnp.asarray(ab), T, Mk, adapt, mc, sk)
            outs.append(jax.tree_util.tree_map(lambda t: np.asarray(t)[:m] if np.ndim(t) and np.shape(t)[0] == bs else np.asarray(t), o))
        res = {}
        for k in outs[0]:
            if isinstance(outs[0][k], dict):
                res[k] = {kk: np.concatenate([o[k][kk] for o in outs]) for kk in outs[0][k]}
            elif k in ('P', 'tau', 'P_text'):
                res[k] = outs[0][k]
            else:
                res[k] = np.concatenate([o[k] for o in outs])
        return res

    def accuracy(self, params, idx, mc=1):
        x = {k: v[idx] for k, v in self.X0.items()}
        o = self.predict(params, x, mc=mc)
        y = self.cls_map[self.bank.Y[idx]]
        return float((o['logits'].argmax(1) == y).mean())


# ------------------------------------------------------------------ simple supervised baselines

class SupTrainer:
    """Unimodal and fusion baselines trained with cross-entropy (same augmentation and dropout policy)."""

    def __init__(self, kind, bank, seen, seed, epochs=60):
        self.kind, self.bank, self.seen, self.seed, self.epochs = kind, bank, np.asarray(seen), seed, epochs
        n = len(seen)
        self.model = {'stgcn': M.STGCNLite(n), 'dcl': M.DeepConvLSTM(n), 'depth': M.DepthTCN(n),
                      'early': M.EarlyFusion(n), 'attn': M.AttnFusion(n), 'edl': M.EvidentialXM(n)}[kind]
        self.uni = {'stgcn': 's', 'dcl': 'u', 'depth': 'v'}.get(kind)

    def _apply(self, params, x, a, train, key):
        if self.uni:
            return self.model.apply({'params': params}, x[self.uni], train=train, rngs={'dropout': key})
        return self.model.apply({'params': params}, x, a, train=train, rngs={'dropout': key})

    def fit(self, tr, va, log=False):
        bank = self.bank
        rng = np.random.RandomState(self.seed)
        self.norm = Norm(bank, tr)
        self.X0 = self.norm(bank.batch(np.arange(bank.N)))
        cls_map = -np.ones(Dt.NCLS, np.int32); cls_map[self.seen] = np.arange(len(self.seen)); self.cls_map = cls_map
        key = jax.random.PRNGKey(self.seed)
        xb = to_j({k: v[:2] for k, v in self.X0.items()})
        if self.uni:
            params = self.model.init({'params': key, 'dropout': key}, xb[self.uni])['params']
        else:
            params = self.model.init({'params': key, 'dropout': key}, xb, jnp.ones((2, 3)))['params']
        steps = self.epochs * (len(tr) // BS)
        opt = optax.chain(optax.clip_by_global_norm(1.0),
                          optax.adamw(optax.warmup_cosine_decay_schedule(0, 1e-3, max(1, min(100, steps // 10)), steps), weight_decay=1e-4))
        st = opt.init(params)
        edl = self.kind == 'edl'
        K = len(self.seen)

        def lossf(params, x, a, y, key, ep):
            o = self._apply(params, x, a, True, key)
            if edl:   # evidential loss (Sensoy et al.) with annealed KL term
                alpha = o + 1.0
                Sa = alpha.sum(1, keepdims=True)
                oh = jax.nn.one_hot(y, K)
                l = (oh * (jnp.log(Sa) - jnp.log(alpha))).sum(1)
                at = (1 - oh) * alpha + oh
                kl = (jax.scipy.special.gammaln(at.sum(1)) - jax.scipy.special.gammaln(K)
                      - jax.scipy.special.gammaln(at).sum(1)
                      + ((at - 1) * (jax.scipy.special.digamma(at) - jax.scipy.special.digamma(at.sum(1, keepdims=True)))).sum(1))
                return (l + jnp.minimum(1.0, ep / 10.0) * kl).mean()
            return optax.softmax_cross_entropy_with_integer_labels(o, y).mean()

        @jax.jit
        def step(params, st, x, a, y, key, ep):
            l, g = jax.value_and_grad(lossf)(params, x, a, y, key, ep)
            up, st = opt.update(g, st, params)
            return optax.apply_updates(params, up), st, l
        best, best_p = -1, params
        tr_ = tr
        for ep in range(self.epochs):
            perm = rng.permutation(tr_)
            for b in range(len(perm) // BS):
                ids = perm[b * BS:(b + 1) * BS]
                x = Dt.augment({k: v[ids] for k, v in self.X0.items()}, rng, p_corrupt=0.0 if self.uni else 0.3)
                a = np.ones((BS, 3), np.float32) if self.uni else drop_mask(BS, rng, float(__import__('os').environ.get('P_DROP', 0.25)))
                if not self.uni:
                    x = {m: x[m] * a[:, i].reshape((-1,) + (1,) * (x[m].ndim - 1)) for i, m in enumerate(Dt.MODS)}
                key, sk = jax.random.split(key)
                params, st, l = step(params, st, to_j(x), jnp.asarray(a), jnp.asarray(cls_map[bank.Y[ids]]), sk,
                                     float(ep))
            if ep >= self.epochs // 3 and (ep % 3 == 2 or ep == self.epochs - 1):
                acc = self.accuracy(params, va)
                if acc >= best:
                    best, best_p = acc, params
                if log:
                    print(f'  ep{ep} {float(l):.3f} val {acc:.3f}', flush=True)
        self.params = best_p
        return best_p

    @functools.partial(jax.jit, static_argnums=(0,))
    def _out(self, params, x, a):
        return self._apply(params, x, a, False, jax.random.PRNGKey(0))

    def predict(self, params, x, a=None, bs=64):
        n = x['s'].shape[0]
        if a is None:
            a = np.ones((n, 3), np.float32)
        outs = []
        for b in range(0, n, bs):
            sl = slice(b, min(n, b + bs)); m = sl.stop - sl.start
            xb = {k: v[sl] for k, v in x.items()}; ab = a[sl]
            if not self.uni:
                xb = {mm: xb[mm] * ab[:, i].reshape((-1,) + (1,) * (xb[mm].ndim - 1)) for i, mm in enumerate(Dt.MODS)}
            if m < bs:
                xb = {k: np.concatenate([v, np.repeat(v[-1:], bs - m, 0)]) for k, v in xb.items()}
                ab = np.concatenate([ab, np.repeat(ab[-1:], bs - m, 0)])
            outs.append(np.asarray(self._out(params, to_j(xb), jnp.asarray(ab)))[:m])
        return np.concatenate(outs)

    def accuracy(self, params, idx):
        o = self.predict(params, {k: v[idx] for k, v in self.X0.items()})
        return float((o.argmax(1) == self.cls_map[self.bank.Y[idx]]).mean())


def n_params(p):
    return int(sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(p)))
