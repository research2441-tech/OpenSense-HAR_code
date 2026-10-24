"""OpenSense-HAR and baseline networks (JAX / Flax linen)."""
import jax, jax.numpy as jnp
import flax.linen as nn
import numpy as np

D = 64          # token width
DP = 128        # shared activity-semantic space width
P1 = 16         # primitive tokens per modality
HEADS = 4
MODS = ('v', 's', 'u')

# Kinect v2 bone list for the GCN baseline
BONES = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22), (20, 8), (8, 9),
         (9, 10), (10, 11), (11, 23), (11, 24), (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17),
         (17, 18), (18, 19)]


def l2n(x, axis=-1):
    return x * jax.lax.rsqrt((x * x).sum(axis, keepdims=True) + 1e-8)


class Block(nn.Module):
    dim: int = D
    drop: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, train=False):
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(num_heads=HEADS, qkv_features=self.dim,
                                            dropout_rate=self.drop, deterministic=not train)(h, h, mask=mask)
        x = x + nn.Dropout(self.drop, deterministic=not train)(h)
        h = nn.LayerNorm()(x)
        h = nn.Dense(2 * self.dim)(h); h = nn.gelu(h); h = nn.Dense(self.dim)(h)
        return x + nn.Dropout(self.drop, deterministic=not train)(h)


class BiasBlock(nn.Module):
    """Pre-LN Transformer block whose attention logits receive an additive per-key bias
    (log reliability of the key's modality); masked keys receive -1e9."""
    dim: int = D
    drop: float = 0.1

    @nn.compact
    def __call__(self, x, kbias, train=False):          # x (B,N,D), kbias (B,N)
        B, N, _ = x.shape
        dh = self.dim // HEADS
        h = nn.LayerNorm()(x)
        q = nn.Dense(self.dim)(h).reshape(B, N, HEADS, dh)
        k = nn.Dense(self.dim)(h).reshape(B, N, HEADS, dh)
        v = nn.Dense(self.dim)(h).reshape(B, N, HEADS, dh)
        lg = jnp.einsum('bqhd,bkhd->bhqk', q, k) / np.sqrt(dh) + kbias[:, None, None, :]
        att = jax.nn.softmax(lg, -1)
        att = nn.Dropout(self.drop, deterministic=not train)(att)
        o = jnp.einsum('bhqk,bkhd->bqhd', att, v).reshape(B, N, self.dim)
        x = x + nn.Dropout(self.drop, deterministic=not train)(nn.Dense(self.dim)(o))
        h = nn.LayerNorm()(x)
        h = nn.Dense(2 * self.dim)(h); h = nn.gelu(h); h = nn.Dense(self.dim)(h)
        return x + nn.Dropout(self.drop, deterministic=not train)(h)


# ------------------------------------------------------------------ stems (primitive tokenisers)

class StemS(nn.Module):
    @nn.compact
    def __call__(self, x):                       # (B,64,150)
        B = x.shape[0]
        return nn.Dense(D)(x.reshape(B, P1, -1))


class StemU(nn.Module):
    @nn.compact
    def __call__(self, x):                       # (B,128,60)
        B = x.shape[0]
        return nn.Dense(D)(x.reshape(B, P1, -1))


class FrameCNN(nn.Module):
    out: int = D

    @nn.compact
    def __call__(self, x):                       # (B,T,32,32,2)
        B, T = x.shape[:2]
        h = x.reshape(B * T, 32, 32, 2)
        h = nn.gelu(nn.Conv(8, (3, 3), strides=(2, 2))(h))
        h = nn.gelu(nn.Conv(16, (3, 3), strides=(2, 2))(h))
        h = nn.gelu(nn.Conv(32, (3, 3), strides=(2, 2))(h))      # 4x4x32
        return nn.Dense(self.out)(h.reshape(B, T, -1))


class StemV(nn.Module):
    @nn.compact
    def __call__(self, x):                       # (B,16,32,32,2)
        return nn.Dense(D)(FrameCNN(D)(x))        # one primitive token per depth frame (B,16,D)


def raw_targets(x):
    """Raw primitive-pair patches used as masked-modelling targets (level-2 granularity: 8 per modality)."""
    B = x['s'].shape[0]
    ts = x['s'].reshape(B, 8, -1)
    tu = x['u'].reshape(B, 8, -1)
    v = x['v'][..., 0].reshape(B, 16, 16, 2, 16, 2).mean((3, 5))      # (B,16,16,16)
    tv = v.reshape(B, 8, -1)
    return {'v': tv, 's': ts, 'u': tu}


# ------------------------------------------------------------------ hierarchical temporal encoder

def local_mask(n, w):
    i = np.arange(n)
    return jnp.asarray((i[:, None] // w) == (i[None, :] // w))


class HierEncoder(nn.Module):
    hier: bool = True

    @nn.compact
    def __call__(self, tok, train=False):         # tok (B,16,D) -> tokens (B,8,D), pooled (B,D)
        if self.hier:
            m = local_mask(P1, 4)[None, None]
            for _ in range(2):
                tok = Block()(tok, mask=m, train=train)       # motion-primitive level (0.5 s windows)
            B = tok.shape[0]
            tok = nn.Dense(D)(tok.reshape(B, P1 // 2, 2 * D))  # merge adjacent primitives
            for _ in range(2):
                tok = Block()(tok, train=train)              # behaviour level (global)
        else:
            for _ in range(4):
                tok = Block()(tok, train=train)
            B = tok.shape[0]
            tok = nn.Dense(D)(tok.reshape(B, P1 // 2, 2 * D))
        tok = nn.LayerNorm()(tok)
        return tok, tok.mean(1)


class TextEncoder(nn.Module):
    @nn.compact
    def __call__(self, T, M):                    # T (...,L,300) frozen GloVe tokens, M (...,L)
        h = nn.gelu(nn.Dense(D)(T))
        q = self.param('q', nn.initializers.normal(0.02), (D,))
        a = jnp.einsum('...ld,d->...l', h, q) / np.sqrt(D)
        a = jnp.where(M > 0, a, -1e9)
        a = jax.nn.softmax(a, -1)
        pooled = jnp.einsum('...l,...ld->...d', a, h)
        mean = (T * M[..., None]).sum(-2) / (M.sum(-1, keepdims=True) + 1e-6)
        h2 = jnp.concatenate([pooled, nn.Dense(D)(mean)], -1)
        h2 = nn.Dense(DP)(nn.gelu(nn.Dense(2 * D)(h2)))
        return l2n(h2)


class Proj(nn.Module):
    out: int = DP

    @nn.compact
    def __call__(self, z, train=False):
        h = nn.Dropout(0.2, deterministic=not train)(z)
        h = nn.gelu(nn.Dense(2 * D)(h))
        h = nn.Dropout(0.2, deterministic=not train)(h)
        return nn.Dense(self.out)(h)


class LatentGen(nn.Module):
    """Conditional Gaussian generator of a modality's pooled latent from the other modalities."""

    @nn.compact
    def __call__(self, zs, avail):               # zs (B,2,D) other-modality latents, avail (B,2)
        h = jnp.concatenate([(zs * avail[..., None]).reshape(zs.shape[0], -1), avail], -1)
        h = nn.gelu(nn.Dense(2 * D)(h))
        h = nn.gelu(nn.Dense(2 * D)(h))
        mu = nn.Dense(D)(h)
        logv = jnp.clip(nn.Dense(D)(h), -6.0, 3.0)
        return mu, logv


class RawGen(nn.Module):
    """Ablation: reconstruct the raw primitive tokens of the missing modality (input space)."""

    @nn.compact
    def __call__(self, zs, avail):
        h = jnp.concatenate([(zs * avail[..., None]).reshape(zs.shape[0], -1), avail], -1)
        h = nn.gelu(nn.Dense(2 * D)(h))
        return nn.Dense(P1 * D)(h).reshape(zs.shape[0], P1, D)


class Adapter(nn.Module):
    r: int = 8

    @nn.compact
    def __call__(self, z):
        a = nn.Dense(self.r, use_bias=False, kernel_init=nn.initializers.normal(0.02), name='A')(z)
        return z + nn.Dense(D, use_bias=False, kernel_init=nn.initializers.zeros, name='B')(a)


# ------------------------------------------------------------------ OpenSense-HAR

class OSHAR(nn.Module):
    """Configurable model.  Flags allow the ablations and the closest baselines to share code.
    lang:  prototype (text) classifier vs. linear head
    rel:   uncertainty-weighted product-of-experts fusion vs. uniform average
    gen:   'latent' | 'raw' | 'feat' | 'none' handling of absent / unreliable modalities
    experts: which embeddings enter fusion ('vsuj' = three modalities + joint cross-modal expert)
    """
    lang: bool = True
    rel: bool = True
    gen: str = 'latent'
    hier: bool = True
    experts: str = 'vsuj'
    ncls: int = 22
    gate_k: float = 4.0
    fusion: str = 'rba'          # 'rba' reliability-biased cross-modal attention | 'poe' late product of experts
    resid: bool = False          # learned residual on seen-class text prototypes
    dual: bool = False           # decoupled language grounding (semantic head) on the expert ensemble

    def setup(self):
        self.stems = {'v': StemV(), 's': StemS(), 'u': StemU()}
        self.pos = self.param('pos', nn.initializers.normal(0.02), (P1, D))
        self.memb = self.param('memb', nn.initializers.normal(0.02), (3, D))
        self.mask_tok = self.param('mask_tok', nn.initializers.normal(0.02), (D,))
        self.enc = HierEncoder(hier=self.hier)
        self.joint = [BiasBlock() for _ in range(2)] if self.fusion == 'rba' else [Block() for _ in range(2)]
        self.sub_proj = nn.Dense(D)
        self.joint_ln = nn.LayerNorm()
        self.dec = {m: nn.Dense(o) for m, o in zip(MODS, (512, 1200, 960))}
        self.proj = {k: Proj() for k in 'vsuj'}
        self.logvar = {k: nn.Dense(1) for k in 'vsuj'}
        self.adapt = {k: Adapter() for k in 'vsu'}
        if self.gen == 'latent':
            self.gens = {m: LatentGen() for m in MODS}
        elif self.gen == 'raw':
            self.gens = {m: RawGen() for m in MODS}
        elif self.gen == 'feat':
            self.gens = {m: LatentGen() for m in MODS}
        self.text = TextEncoder()
        self.proto = self.param('proto', nn.initializers.normal(1.0), (self.ncls, DP))
        self.proto_res = self.param('proto_res', nn.initializers.zeros, (self.ncls, DP))
        self.sem1 = nn.Dense(DP)
        self.sem2 = nn.Dense(DP)
        self.log_tau_s = self.param('log_tau_s', lambda k: jnp.array(np.log(0.07), jnp.float32))
        self.log_tau = self.param('log_tau', lambda k: jnp.array(np.log(0.07), jnp.float32))
        self.gate_c = self.param('gate_c', lambda k: jnp.array(0.0, jnp.float32))
        self.gen_c = self.param('gen_c', lambda k: jnp.array(0.0, jnp.float32))

    # -------------------------------------------------------------- encoding
    def tokens(self, x, pmask=None):
        out = {}
        for i, m in enumerate(MODS):
            t = self.stems[m](x[m]) + self.pos[None] + self.memb[i][None, None]
            if pmask is not None:                 # (B,3,8) pair-level masks -> 16 tokens
                pm = jnp.repeat(pmask[:, i], 2, axis=1)[..., None]
                t = jnp.where(pm > 0, self.mask_tok[None, None], t)
            out[m] = t
        return out

    def encode(self, x, avail, train=False, pmask=None, adapt=False):
        tok = self.tokens(x, pmask)
        B = x['s'].shape[0]
        allt = jnp.concatenate([tok[m] for m in MODS], 0)          # shared encoder in one pass
        H, z = self.enc(allt, train=train)
        Hs = {m: H[i * B:(i + 1) * B] for i, m in enumerate(MODS)}
        zs = {m: z[i * B:(i + 1) * B] for i, m in enumerate(MODS)}
        if adapt:
            zs = {m: self.adapt[m](zs[m]) for m in MODS}
        return Hs, zs

    def joint_expert(self, Hs, avail, train=False, logw=None, subs=None, sub_logw=None):
        Hc = jnp.concatenate([Hs[m] for m in MODS], 1)              # (B,24,D)
        km = jnp.repeat(avail, 8, axis=1)                          # (B,24)
        if self.fusion == 'rba':
            B = Hc.shape[0]
            lw = jnp.zeros_like(avail) if logw is None else logw
            kb = jnp.where(km > 0, jnp.repeat(lw, 8, axis=1), -1e9)
            if subs is not None:
                Hc = jnp.concatenate([Hc, subs], 1)                # (B,27,D) latent substitution tokens
                kb = jnp.concatenate([kb, sub_logw], 1)
                kv = jnp.concatenate([km, (sub_logw > -1e8).astype(km.dtype)], 1)
            else:
                kv = km
            h = Hc
            for b in self.joint:
                h = b(h, kb, train=train)
            h = self.joint_ln(h)
            wq = kv * jnp.exp(jnp.where(kb > -1e8, kb, -30.0))       # reliability-weighted pooling
            zj = (h * wq[..., None]).sum(1) / (wq.sum(1, keepdims=True) + 1e-6)
            return h[:, :24], zj
        mask = (km[:, None, :, None] * km[:, None, None, :]) > 0
        mask = mask | jnp.eye(24, dtype=bool)[None, None]
        h = Hc
        for b in self.joint:
            h = b(h, mask=mask, train=train)
        h = self.joint_ln(h)
        zj = (h * km[..., None]).sum(1) / (km.sum(1, keepdims=True) + 1e-6)
        return h, zj

    def text_protos(self, T, M):
        e = self.text(T, M)                                         # (C,J,DP)
        return l2n(e.mean(-2))                                     # paraphrase-averaged prototype

    # -------------------------------------------------------------- forward
    def __call__(self, x, avail, T, M, train=False, pmask=None, adapt=False, gen_drop=None, mc=1):
        """Returns dict with per-expert embeddings, log-variances, fused embedding and logits.
        avail: (B,3) availability of v,s,u.  gen_drop: optional (B,3) modalities hidden from the
        encoder but whose clean latents serve as generator targets (training)."""
        B = x['s'].shape[0]
        Hs, zs = self.encode(x, avail, train=train, pmask=pmask, adapt=adapt)
        out = {'z': zs, 'H': Hs}
        a = avail
        # ---------------- generator: predict each modality's latent from the others
        gen_mu, gen_lv = {}, {}
        if self.gen in ('latent', 'feat'):
            for i, m in enumerate(MODS):
                oth = [j for j in range(3) if j != i]
                zo = jnp.stack([jax.lax.stop_gradient(zs[MODS[j]]) if self.gen == 'feat' else zs[MODS[j]]
                                for j in oth], 1)
                mu, lv = self.gens[m](zo, a[:, oth])
                gen_mu[m], gen_lv[m] = mu, lv
        out['gen_mu'], out['gen_lv'] = gen_mu, gen_lv
        if self.gen == 'raw':
            raw_tok = {}
            for i, m in enumerate(MODS):
                oth = [j for j in range(3) if j != i]
                zo = jnp.stack([zs[MODS[j]] for j in oth], 1)
                raw_tok[m] = self.gens[m](zo, a[:, oth])
            out['raw_tok'] = raw_tok
        # ---------------- per-modality uncertainty before substitution
        lv_obs = {m: self.logvar[m](zs[m])[:, 0] for m in MODS}
        if self.fusion == 'rba':
            return self._rba(out, Hs, zs, a, lv_obs, gen_mu, gen_lv, T, M, train, mc, B)
        # ---------------- substitute missing / gate unreliable latents
        zt, a_eff, lv_sub = {}, [], {}
        for i, m in enumerate(MODS):
            ai = a[:, i][:, None]
            if self.gen in ('latent', 'feat'):
                if self.gen == 'latent':
                    g = jax.nn.sigmoid(self.gate_k * (lv_obs[m] - self.gate_c))[:, None] if self.rel else 0.0
                else:
                    g = 0.0
                g = g * ai
                zt[m] = ai * ((1 - g) * zs[m] + g * gen_mu[m]) + (1 - ai) * gen_mu[m]
                any_other = (a.sum(1) - a[:, i]) > 0
                a_eff.append(jnp.where(any_other | (a[:, i] > 0), 1.0, 0.0))
                lv_sub[m] = jnp.where(a[:, i] > 0, lv_obs[m],
                                      jnp.log(jnp.exp(self.logvar[m](gen_mu[m])[:, 0]) + jnp.exp(gen_lv[m]).mean(-1)))
            elif self.gen == 'raw':
                # missing modality re-encoded from reconstructed primitive tokens
                rt = out['raw_tok'][m]
                _, zr = self.enc(rt + self.pos[None] + self.memb[i][None, None], train=train)
                zt[m] = ai * zs[m] + (1 - ai) * zr
                any_other = (a.sum(1) - a[:, i]) > 0
                a_eff.append(jnp.where(any_other | (a[:, i] > 0), 1.0, 0.0))
                lv_sub[m] = jnp.where(a[:, i] > 0, lv_obs[m], self.logvar[m](zr)[:, 0])
            else:
                zt[m] = zs[m] * ai
                a_eff.append(a[:, i])
                lv_sub[m] = lv_obs[m]
        a_eff = jnp.stack(a_eff, 1)
        # joint cross-modal expert (only on observed tokens)
        if 'j' in self.experts:
            Hj, zj = self.joint_expert(Hs, a, train=train)
            out['Hj'] = Hj
            zt['j'] = zj
            lv_sub['j'] = self.logvar['j'](zj)[:, 0]
            lv_obs['j'] = lv_sub['j']
        keys = [k for k in 'vsuj' if k in self.experts]
        # ---------------- projections (MC dropout -> epistemic spread)
        embs, epi = {}, {}
        for k in keys:
            if mc > 1:      # MC-dropout through the projection head -> epistemic spread
                smp = jnp.stack([l2n(self.proj[k](zt[k], train=True)) for _ in range(mc)], 0)
                e = l2n(smp.mean(0))
                epi[k] = ((smp - smp.mean(0, keepdims=True)) ** 2).sum(-1).mean(0)
            else:
                e = l2n(self.proj[k](zt[k], train=train))
                epi[k] = jnp.zeros(B)
            embs[k] = e
        out['epi'] = epi
        out['emb'] = embs
        out['lv'] = lv_sub
        out['lv_obs'] = lv_obs
        mk = {'v': a_eff[:, 0], 's': a_eff[:, 1], 'u': a_eff[:, 2], 'j': (a.sum(1) > 0).astype(jnp.float32)}
        out['mk'] = mk
        # ---------------- fusion
        if self.rel:
            pi = jnp.stack([mk[k] / (jnp.exp(lv_sub[k]) + epi[k]) for k in keys], 1)
        else:
            pi = jnp.stack([mk[k] for k in keys], 1)
        w = pi / (pi.sum(1, keepdims=True) + 1e-8)
        out['w'] = w
        ef = l2n(jnp.einsum('bk,bkd->bd', w, jnp.stack([embs[k] for k in keys], 1)))
        out['ef'] = ef
        # fused aleatoric variance of the product of experts
        out['lv_f'] = -jnp.log(pi.sum(1) + 1e-8)
        # ---------------- logits
        tau = jnp.exp(self.log_tau)
        out['tau'] = tau
        if self.dual:
            # decoupled grounding: closed-set decisions on learned prototypes, language alignment in a
            # semantic head attached to the fused embedding (used for retrieval of unseen activities)
            Pc = l2n(self.proto)
            out['P'] = Pc
            out['logits'] = ef @ Pc.T / tau
            out['logits_m'] = {k: embs[k] @ Pc.T / tau for k in keys}
            es = l2n(self.sem2(nn.gelu(self.sem1(ef))))
            out['es'] = es
            if T is not None:
                Pt = self.text_protos(T, M)
                out['P_text'] = Pt
                out['logits_sem'] = es @ Pt.T / jnp.exp(self.log_tau_s)
            return out
        if self.lang:
            P = self.text_protos(T, M)                            # (C,DP)
            out['P'] = P
            out['logits'] = ef @ P.T / tau
            out['logits_m'] = {k: embs[k] @ P.T / tau for k in keys}
        else:   # learned (label-agnostic) cosine prototypes
            P = l2n(self.proto)
            out['P'] = P
            out['logits'] = ef @ P.T / tau
            out['logits_m'] = {k: embs[k] @ P.T / tau for k in keys}
        return out

    def _project(self, k, z, train, mc, B):
        if mc > 1:      # MC-dropout through the projection head -> epistemic spread
            smp = jnp.stack([l2n(self.proj[k](z, train=True)) for _ in range(mc)], 0)
            return l2n(smp.mean(0)), ((smp - smp.mean(0, keepdims=True)) ** 2).sum(-1).mean(0)
        return l2n(self.proj[k](z, train=train)), jnp.zeros(B)

    def _rba(self, out, Hs, zs, a, lv_obs, gen_mu, gen_lv, T, M, train, mc, B):
        """Reliability-biased cross-modal attention with latent substitution tokens."""
        embs, epi = {}, {}
        for m in MODS:
            embs[m], epi[m] = self._project(m, zs[m], train, mc, B)
        # total predictive variance of each observed stream: aleatoric + epistemic
        lvar = jnp.stack([jnp.log(jnp.exp(lv_obs[m]) + epi[m]) for m in MODS], 1)       # (B,3)
        if self.rel:
            lw = jnp.where(a > 0, -lvar, -1e9)
            nav = a.sum(1, keepdims=True)
            logw = jax.nn.log_softmax(lw, axis=1) + jnp.log(nav)                       # 0 when equally reliable
            logw = jnp.where(a > 0, logw, 0.0)
            gate = a * jax.nn.sigmoid(self.gate_k * (lvar - self.gate_c))
        else:
            logw = jnp.zeros_like(a)
            gate = jnp.zeros_like(a)
        subs = None; sub_logw = None
        if self.gen == 'latent':
            subs = jnp.stack([self.sub_proj(gen_mu[m]) + self.memb[i][None] for i, m in enumerate(MODS)], 1)
            any_other = (a.sum(1, keepdims=True) - a) > 0
            # confidence of a substitute for an absent stream from the generator's predictive variance
            sub_on = jnp.where(a > 0, gate, 1.0) * any_other
            sub_logw = jnp.where(sub_on > 1e-3, jnp.log(sub_on + 1e-6), -1e9)
        elif self.gen == 'raw':
            Hs = dict(Hs)
            av = a
            for i, m in enumerate(MODS):
                oth = [j for j in range(3) if j != i]
                zo = jnp.stack([zs[MODS[j]] for j in oth], 1)
                rt = self.gens[m](zo, a[:, oth])
                Hr, _ = self.enc(rt + self.pos[None] + self.memb[i][None, None], train=train)
                Hs[m] = jnp.where(a[:, i][:, None, None] > 0, Hs[m], Hr)
                out.setdefault('raw_tok', {})[m] = rt
            av = jnp.ones_like(a)
            a = av
        Hj, zj = self.joint_expert(Hs, a, train=train, logw=logw, subs=subs, sub_logw=sub_logw)
        out['Hj'] = Hj
        ej, epij = self._project('j', zj, train, mc, B)
        embs['j'] = ej; epi['j'] = epij
        lv_obs = dict(lv_obs); lv_obs['j'] = self.logvar['j'](zj)[:, 0]
        out['emb'] = embs; out['epi'] = epi; out['lv_obs'] = lv_obs; out['lv'] = lv_obs
        out['mk'] = {'v': a[:, 0], 's': a[:, 1], 'u': a[:, 2], 'j': jnp.ones(B)}
        out['w'] = jnp.concatenate([jnp.exp(logw) * a / jnp.maximum(a.sum(1, keepdims=True), 1), gate], 1)
        out['ef'] = ej
        out['lv_f'] = jnp.log(jnp.exp(lv_obs['j']) + epij)
        tau = jnp.exp(self.log_tau); out['tau'] = tau
        P = self.text_protos(T, M) if self.lang else l2n(self.proto)
        out['P_text'] = P
        if self.lang and self.resid and P.shape[0] == self.ncls:
            P = l2n(P + self.proto_res)
        out['P'] = P
        out['logits'] = ej @ P.T / tau
        out['logits_m'] = {k: embs[k] @ P.T / tau for k in 'vsuj'}
        if self.gen == 'latent':
            out['logits_gen'] = {m: self._project(m, gen_mu[m], train, 1, B)[0] @ P.T / tau for m in MODS}
        return out

    def init_all(self, x, avail, T, M):
        out = self(x, avail, T, M, adapt=True)
        if 'Hj' in out:
            self.decode(out['Hj'])
        return out

    def decode(self, Hj):
        """Masked-modelling decoder from cross-modal tokens (B,24,D) -> raw pair patches."""
        return {m: self.dec[m](Hj[:, 8 * i:8 * (i + 1)]) for i, m in enumerate(MODS)}


# ------------------------------------------------------------------ baselines

def kinect_adj():
    A = np.eye(25, dtype=np.float32)
    for i, j in BONES:
        A[i, j] = A[j, i] = 1.0
    d = A.sum(1)
    return jnp.asarray(A / np.sqrt(d[:, None] * d[None, :]))


class STGCNLite(nn.Module):
    """Spatial-temporal graph convolution (ST-GCN style) for the skeleton stream."""
    ncls: int = 22

    @nn.compact
    def __call__(self, x, train=False, return_feat=False):     # (B,64,150)
        B = x.shape[0]
        h = jnp.concatenate([x[..., :75].reshape(B, 64, 25, 3), x[..., 75:].reshape(B, 64, 25, 3)], -1)
        A = kinect_adj()
        for c, s in ((32, 1), (64, 2), (64, 1), (96, 2)):
            h = jnp.einsum('vw,btwc->btvc', A, h)
            h = nn.Dense(c)(h)
            h = nn.gelu(nn.LayerNorm()(h))
            h = nn.Conv(c, (9, 1), strides=(s, 1), padding='SAME')(h)
            h = nn.gelu(nn.LayerNorm()(h))
            h = nn.Dropout(0.1, deterministic=not train)(h)
        f = h.mean((1, 2))
        return f if return_feat else nn.Dense(self.ncls)(f)


class DeepConvLSTM(nn.Module):
    ncls: int = 22

    @nn.compact
    def __call__(self, x, train=False, return_feat=False):     # (B,128,60)
        h = x
        for _ in range(4):
            h = nn.gelu(nn.Conv(64, (5,), padding='SAME')(h))
        h = nn.Dropout(0.2, deterministic=not train)(h)
        h = nn.RNN(nn.OptimizedLSTMCell(96))(h)
        h = nn.RNN(nn.OptimizedLSTMCell(96))(h)
        f = h[:, -1]
        return f if return_feat else nn.Dense(self.ncls)(f)


class DepthTCN(nn.Module):
    """Frame CNN + temporal convolution for the depth stream."""
    ncls: int = 22

    @nn.compact
    def __call__(self, x, train=False, return_feat=False):     # (B,16,32,32,2)
        f = FrameCNN(D)(x)
        for c in (96, 96):
            f = nn.gelu(nn.Conv(c, (5,), padding='SAME')(f))
        f = nn.Dropout(0.2, deterministic=not train)(f).mean(1)
        return f if return_feat else nn.Dense(self.ncls)(f)


class EarlyFusion(nn.Module):
    """Feature-level fusion: time-aligned primitive tokens concatenated, then a flat Transformer."""
    ncls: int = 22

    @nn.compact
    def __call__(self, x, avail, train=False):
        t = [StemV()(x['v']), StemS()(x['s']), StemU()(x['u'])]
        t = [ti * avail[:, i][:, None, None] for i, ti in enumerate(t)]
        h = nn.Dense(D)(jnp.concatenate(t, -1))
        h = h + self.param('pos', nn.initializers.normal(0.02), (P1, D))[None]
        for _ in range(4):
            h = Block()(h, train=train)
        return nn.Dense(self.ncls)(nn.LayerNorm()(h).mean(1))


class AttnFusion(nn.Module):
    """HAMLET-style: unimodal encoders, then multi-head attention across modality summaries."""
    ncls: int = 22

    @nn.compact
    def __call__(self, x, avail, train=False):
        pos = self.param('pos', nn.initializers.normal(0.02), (P1, D))
        memb = self.param('memb', nn.initializers.normal(0.02), (3, D))
        enc = HierEncoder()
        stems = [StemV(), StemS(), StemU()]
        zs = []
        for i, m in enumerate(MODS):
            _, z = enc(stems[i](x[m]) + pos[None] + memb[i][None, None], train=train)
            zs.append(z)
        Z = jnp.stack(zs, 1)                                      # (B,3,D)
        mask = (avail[:, None, :, None] * avail[:, None, None, :]) > 0
        mask = mask | jnp.eye(3, dtype=bool)[None, None]
        h = Block()(Z, mask=mask, train=train)
        f = (h * avail[..., None]).sum(1) / (avail.sum(1, keepdims=True) + 1e-6)
        return nn.Dense(self.ncls)(f)


class EvidentialXM(nn.Module):
    """Cross-modal Transformer with a Dirichlet evidential head (DEAR-style open-set baseline)."""
    ncls: int = 22

    @nn.compact
    def __call__(self, x, avail, train=False):
        m = OSHAR(lang=False, rel=False, gen='none', experts='j', ncls=self.ncls, fusion='poe', name='core')
        o = m(x, avail, None, None, train=train)
        return jax.nn.softplus(o['logits'] * o['tau'] * 10.0)    # evidence from cosine scores
