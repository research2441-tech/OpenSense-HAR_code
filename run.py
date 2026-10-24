"""Experiment runner. One call = one (experiment, configuration, test subject, seed).

python3 run.py closed opensense 0 0
"""
import sys, os, time, pickle, json
import numpy as np
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import jax
import data as Dt
import train as TR

CONFIGS = {
    # proposed: reliability-weighted expert ensemble + uncertainty-gated latent substitution +
    # masked multimodal pretraining + decoupled language-grounded semantic head
    'opensense': dict(kind='os', fusion='poe', rel=True, gen='latent', dual=True),
    # closest baselines sharing the encoder family
    'xmtf': dict(kind='os', lang=False, rel=False, gen='none', experts='j', mm_pretrain=False, fusion='poe'),
    'actionmae': dict(kind='os', lang=False, rel=False, gen='feat', experts='vsuj', mm_pretrain=False, fusion='poe'),
    'langbind': dict(kind='os', lang=True, rel=False, gen='none', experts='vsu', mm_pretrain=False, fusion='poe'),
    # supervised baselines
    'stgcn': dict(kind='sup'), 'dcl': dict(kind='sup'), 'depth': dict(kind='sup'),
    'early': dict(kind='sup'), 'attn': dict(kind='sup'), 'edl': dict(kind='sup'),
    # ablations of the proposed model
    'abl_nopre': dict(kind='os', fusion='poe', rel=True, gen='latent', dual=True, mm_pretrain=False),
    'abl_coupled': dict(kind='os', fusion='poe', rel=True, gen='latent', dual=False, lang=True),
    'abl_norel': dict(kind='os', fusion='poe', rel=False, gen='latent', dual=True),
    'abl_gennone': dict(kind='os', fusion='poe', rel=True, gen='none', dual=True),
    'abl_genraw': dict(kind='os', fusion='poe', rel=True, gen='raw', dual=True),
    'abl_flat': dict(kind='os', fusion='poe', rel=True, gen='latent', dual=True, hier=False),
    'abl_rba': dict(kind='os', fusion='rba', rel=True, gen='latent'),
    'abl_labeltext': dict(kind='os', fusion='poe', rel=True, gen='latent', dual=True, text='label'),
    # design-stage variants (development split only)
    'dual_lat': dict(kind='os', fusion='poe', rel=False, gen='latent', dual=True),
    'dual_feat_nopre': dict(kind='os', fusion='poe', rel=False, gen='feat', dual=True, mm_pretrain=False),
    'dual_lat_rel': dict(kind='os', fusion='poe', rel=True, gen='latent', dual=True),
    'poe_norel': dict(kind='os', fusion='poe', rel=False),
    'actionmae_lang': dict(kind='os', lang=True, rel=False, gen='feat', experts='vsuj', mm_pretrain=False, fusion='poe'),
}

CORR = [('u', 'noise'), ('u', 'occlude'), ('u', 'drift'), ('u', 'clip'),
        ('s', 'noise'), ('s', 'occlude'), ('s', 'drop'),
        ('v', 'noise'), ('v', 'occlude'), ('v', 'holes')]
MISSING = ['v', 's', 'u', 'vs', 'vu', 'su']


def build(cfg_name, bank, seen, seed):
    c = dict(CONFIGS[cfg_name])
    kind = c.pop('kind')
    ep = int(os.environ.get('EPOCHS', 40)); pre = int(os.environ.get('PRE_EPOCHS', 20))
    if kind == 'sup':
        return TR.SupTrainer(cfg_name, bank, seen, seed, epochs=ep), 'sup'
    text = c.pop('text', 'desc')
    c.setdefault('epochs', ep); c.setdefault('pre_epochs', pre)
    return TR.OSTrainer(c, bank, seen, seed, text=text), 'os'


def outputs(tr, kind, x, a=None, classes=None):
    if kind == 'sup':
        lg = tr.predict(tr.params, x, a)
        return {'logits': lg.astype(np.float32)}
    o = tr.predict(tr.params, x, a, classes=classes, mc=8)
    r = {'logits': o['logits'].astype(np.float32), 'ef': o['ef'].astype(np.float16),
         'lv_f': o['lv_f'].astype(np.float32), 'w': o['w'].astype(np.float32)}
    if 'es' in o:
        r['es'] = o['es'].astype(np.float16)
    return r


def main():
    exp, cfg_name, fold, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    extra = sys.argv[5] if len(sys.argv) > 5 else None
    bank = Dt.Bank()
    outdir = f'results/{exp}'; os.makedirs(outdir, exist_ok=True)
    tag = f'{cfg_name}_f{fold}_s{seed}' + (f'_{extra}' if extra else '')
    if os.path.exists(f'{outdir}/{tag}.pkl'):
        print('exists', tag); return
    if os.path.exists('skip.txt') and ' '.join(sys.argv[1:]) in open('skip.txt').read().splitlines():
        print('skipped', tag); return
    t0 = time.time()
    if exp == 'dev':     # development split used only for design decisions: train cyy/myj/qyh, test zyh
        bank.SUB = bank.SUB.copy()
        keep = bank.SUB != 0
        seen = np.arange(Dt.NCLS)
        tr_idx, va_idx, te_idx = Dt.loso(bank, int(extra) if extra else 4, seed)
        tr_idx = tr_idx[keep[tr_idx]]; va_idx = va_idx[keep[va_idx]]
        trn, kind = build(cfg_name, bank, seen, seed)
        trn.fit(tr_idx, va_idx)
        X = trn.X0; xt = {k: v[te_idx] for k, v in X.items()}; n = len(te_idx)
        res = {'y': bank.Y[te_idx], 'clean': outputs(trn, kind, xt)}
        for pat in MISSING:
            res['avail_' + pat] = outputs(trn, kind, xt, Dt.mask_dict(n, pat))
        for (m, k) in CORR:
            res[f'cor_{m}_{k}_3'] = outputs(trn, kind, Dt.corrupt_batch(xt, m, k, 3, seed=7))
        for yaw in (30, 60):
            res[f'yaw_{yaw}'] = outputs(trn, kind, trn.norm(bank.batch(te_idx, yaw)))
        pickle.dump(res, open(f'{outdir}/{tag}.pkl', 'wb')); print(tag, 'done', round(time.time() - t0)); return
    if exp in ('closed', 'limited'):
        seen = np.arange(Dt.NCLS)
        tr_idx, va_idx, te_idx = Dt.loso(bank, fold, seed)
        trn, kind = build(cfg_name, bank, seen, seed)
        if exp == 'limited':
            frac = float(extra)
            trn.fit(tr_idx, va_idx, label_frac=frac) if kind == 'os' else None
        else:
            trn.fit(tr_idx, va_idx)
        ttrain = time.time() - t0
        res = {'te': te_idx, 'y': bank.Y[te_idx], 'val_acc': getattr(trn, 'val_acc', None), 'train_s': ttrain}
        X = trn.X0
        xt = {k: v[te_idx] for k, v in X.items()}
        n = len(te_idx)
        t1 = time.time()
        res['clean'] = outputs(trn, kind, xt)
        res['infer_s_per_sample'] = (time.time() - t1) / n
        if exp == 'closed':
            for pat in MISSING:
                res['avail_' + pat] = outputs(trn, kind, xt, Dt.mask_dict(n, pat))
            for (m, k) in CORR:
                for sev in (1, 3, 5):
                    xc = Dt.corrupt_batch(xt, m, k, sev, seed=100 * sev + fold)
                    res[f'cor_{m}_{k}_{sev}'] = outputs(trn, kind, xc)
            for yaw in (30, 60):
                xv = trn.norm(bank.batch(te_idx, yaw))
                res[f'yaw_{yaw}'] = outputs(trn, kind, xv)
            if cfg_name == 'opensense' or cfg_name.startswith('abl_'):
                res['val'] = outputs(trn, kind, {k: v[va_idx] for k, v in X.items()})
                res['va'] = va_idx
        res['n_params'] = TR.n_params(trn.params)
        if cfg_name in ('opensense', 'abl_norel') and exp == 'closed':
            pickle.dump({'params': jax.device_get(trn.params), 'norm': (trn.norm.mu, trn.norm.sd)},
                        open(f'{outdir}/params_{tag}.pkl', 'wb'))
    elif exp == 'open':
        split = extra
        seen, desc, unk = Dt.class_split(split)
        tr_idx, va_idx, te_idx = Dt.loso(bank, fold, seed, classes=seen)
        trn, kind = build(cfg_name, bank, seen, seed)
        trn.fit(tr_idx, va_idx)
        res = {'te': te_idx, 'va': va_idx, 'tr': tr_idx, 'y_te': bank.Y[te_idx], 'y_va': bank.Y[va_idx],
               'y_tr': bank.Y[tr_idx], 'seen': seen, 'desc': desc, 'unk': unk, 'train_s': time.time() - t0}
        X = trn.X0
        for nm, idx in (('tr', tr_idx), ('va', va_idx), ('te', te_idx)):
            x = {k: v[idx] for k, v in X.items()}
            if kind == 'sup':
                res['out_' + nm] = outputs(trn, kind, x)
            else:
                res['out_' + nm] = outputs(trn, kind, x, classes=np.arange(Dt.NCLS))
        if kind == 'os':
            # text prototypes for all classes (descriptions of unknown classes are never used for decisions)
            o = trn.predict(trn.params, {k: v[te_idx[:2]] for k, v in X.items()}, classes=np.arange(Dt.NCLS), mc=1)
            res['P_all'] = o['P_text'] if 'P_text' in o else o['P']; res['tau'] = float(o['tau'])
            res['dual'] = 'P_text' in o
            # missing-modality open-world test (skeleton absent, inertial absent)
            xt = {k: v[te_idx] for k, v in X.items()}
            for pat in ('vu', 'vs'):
                res['te_avail_' + pat] = outputs(trn, kind, xt, Dt.mask_dict(len(te_idx), pat), classes=np.arange(Dt.NCLS))
        if cfg_name == 'opensense' and split == 'mirror':
            pickle.dump({'params': jax.device_get(trn.params), 'norm': (trn.norm.mu, trn.norm.sd)},
                        open(f'{outdir}/params_{tag}.pkl', 'wb'))
    pickle.dump(res, open(f'{outdir}/{tag}.pkl', 'wb'))
    print(tag, 'done', round(time.time() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    main()
