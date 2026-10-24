"""Aggregate all experiment outputs into numbers (results/summary.json) used by tables, figures and text."""
import pickle, glob, json, os, itertools
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

R = 'results'
MAIN = ['stgcn', 'dcl', 'depth', 'late', 'early', 'attn', 'xmtf', 'actionmae', 'langbind', 'opensense']
FUSION = ['late', 'early', 'attn', 'xmtf', 'actionmae', 'langbind', 'opensense']
ABL = ['opensense', 'abl_nopre', 'abl_coupled', 'abl_norel', 'abl_gennone', 'abl_genraw', 'abl_flat']
PATS = ['vs', 'vu', 'su', 'v', 's', 'u']
CORR = [('u', 'noise'), ('u', 'occlude'), ('u', 'drift'), ('u', 'clip'), ('s', 'noise'), ('s', 'occlude'), ('s', 'drop'),
        ('v', 'noise'), ('v', 'occlude'), ('v', 'holes')]
rng = np.random.RandomState(0)


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def load_closed(cfg, seed=0):
    out = []
    for f in range(5):
        p = f'{R}/closed/{cfg}_f{f}_s{seed}.pkl'
        if not os.path.exists(p):
            return None
        out.append(pickle.load(open(p, 'rb')))
    return out


def pooled(runs, key):
    """Concatenate predictions (probabilities) over the 5 LOSO folds."""
    P = np.concatenate([softmax(r[key]['logits']) for r in runs])
    y = np.concatenate([r['y'] for r in runs])
    return P, y


def late_fusion(uni, key, pat=None):
    """Average posteriors of the unimodal nets over the available streams."""
    mods = {'depth': 'v', 'stgcn': 's', 'dcl': 'u'}
    ref = 'clean' if (key.startswith('avail') or key == 'clean') else key
    Ps = []
    for cfg, m in mods.items():
        if pat is not None and m not in pat:
            continue
        k = ref
        if key.startswith('cor_') and key.split('_')[1] != m:
            k = 'clean'
        if key.startswith('yaw') and m == 'u':
            k = 'clean'
        P, y = pooled(uni[cfg], k if k in uni[cfg][0] else 'clean')
        Ps.append(P)
    return np.mean(Ps, 0), y


def get_probs(name, runs, key, uni=None):
    if name == 'late':
        pat = key.split('_')[1] if key.startswith('avail') else None
        return late_fusion(uni, key, pat)
    if name in ('stgcn', 'dcl', 'depth') and key.startswith('avail'):
        m = {'depth': 'v', 'stgcn': 's', 'dcl': 'u'}[name]
        P, y = pooled(runs, 'clean')
        if m not in key.split('_')[1]:
            P = np.full_like(P, 1.0 / P.shape[1])     # stream absent: chance-level output
        return P, y
    return pooled(runs, key)


def acc_ci(correct, B=2000):
    n = len(correct)
    bs = [correct[rng.randint(0, n, n)].mean() for _ in range(B)]
    return float(correct.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ece(P, y, M=15):
    conf = P.max(1); pred = P.argmax(1); cor = (pred == y)
    edges = np.linspace(0, 1, M + 1); e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(cor[m].mean() - conf[m].mean())
    return float(e)


def mcnemar(c1, c2):
    b = int((c1 & ~c2).sum()); c = int((~c1 & c2).sum())
    if b + c == 0:
        return 1.0, b, c
    return float(stats.binomtest(min(b, c), b + c, 0.5).pvalue), b, c


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj.tolist()


def closed_summary():
    S = {}
    runs = {c: load_closed(c) for c in MAIN if c != 'late'}
    runs.update({c: load_closed(c) for c in ABL if c not in runs})
    uni = {c: runs[c] for c in ('stgcn', 'dcl', 'depth')}
    have = [c for c in MAIN if c == 'late' and all(uni[u] is not None for u in uni) or (c != 'late' and runs.get(c))]
    keys = ['clean'] + ['avail_' + p for p in PATS] + [f'cor_{m}_{k}_{s}' for m, k in CORR for s in (1, 3, 5)] + ['yaw_30', 'yaw_60']
    acc = {}
    corr_vec = {}
    for c in have + [a for a in ABL if runs.get(a) and a not in have]:
        acc[c] = {}
        for k in keys:
            P, y = get_probs(c, runs.get(c), k, uni)
            cor = P.argmax(1) == y
            acc[c][k] = float(cor.mean())
            corr_vec[(c, k)] = cor
        P, y = get_probs(c, runs.get(c), 'clean', uni)
        acc[c]['f1'] = float(f1_score(y, P.argmax(1), average='macro'))
        acc[c]['ece'] = ece(P, y)
        a, lo, hi = acc_ci(corr_vec[(c, 'clean')]); acc[c]['clean_ci'] = [lo, hi]
        acc[c]['miss_mean'] = float(np.mean([acc[c]['avail_' + p] for p in PATS]))
        acc[c]['cor_mean'] = float(np.mean([acc[c][f'cor_{m}_{k}_{s}'] for m, k in CORR for s in (1, 3, 5)]))
        for m in 'vsu':
            for s in (1, 3, 5):
                acc[c][f'cor_{m}_sev{s}'] = float(np.mean([acc[c][f'cor_{mm}_{k}_{s}'] for mm, k in CORR if mm == m]))
        acc[c]['yaw_mean'] = float(np.mean([acc[c]['yaw_30'], acc[c]['yaw_60']]))
        P, y = get_probs(c, runs.get(c), 'cor_u_occlude_5', uni)
        acc[c]['ece_cor'] = float(np.mean([ece(*get_probs(c, runs.get(c), f'cor_{m}_{k}_3', uni)) for m, k in CORR]))
        if c in runs and runs[c] is not None:
            acc[c]['params'] = runs[c][0]['n_params']
            acc[c]['train_s'] = float(np.mean([r['train_s'] for r in runs[c]]))
            acc[c]['latency_ms'] = float(np.mean([r['infer_s_per_sample'] for r in runs[c]]) * 1000)
    S['acc'] = acc
    # McNemar vs OpenSense on clean, mean-missing pooled, yaw60
    tests = {}
    for k in ['clean', 'avail_v', 'avail_s', 'avail_u', 'yaw_60', 'cor_u_occlude_5', 'cor_s_occlude_5']:
        ps, rows = [], []
        for c in have:
            if c == 'opensense':
                continue
            p, b, cc = mcnemar(corr_vec[('opensense', k)], corr_vec[(c, k)])
            ps.append(p); rows.append((c, b, cc))
        adj = holm(ps)
        tests[k] = {c: {'p': p, 'p_holm': a, 'os_only': b, 'base_only': cc} for (c, b, cc), p, a in zip(rows, ps, adj)}
    # pooled "all incomplete patterns" test: concatenate the six patterns
    ps, rows = [], []
    for c in have:
        if c == 'opensense':
            continue
        c1 = np.concatenate([corr_vec[('opensense', 'avail_' + p)] for p in PATS])
        c2 = np.concatenate([corr_vec[(c, 'avail_' + p)] for p in PATS])
        p, b, cc = mcnemar(c1, c2); ps.append(p); rows.append((c, b, cc))
    tests['missing_all'] = {c: {'p': p, 'p_holm': a, 'os_only': b, 'base_only': cc} for (c, b, cc), p, a in zip(rows, ps, holm(ps))}
    ps, rows = [], []
    for c in have:
        if c == 'opensense':
            continue
        ck = [f'cor_{m}_{k}_{s}' for m, k in CORR for s in (1, 3, 5)]
        c1 = np.concatenate([corr_vec[('opensense', k)] for k in ck]); c2 = np.concatenate([corr_vec[(c, k)] for k in ck])
        p, b, cc = mcnemar(c1, c2); ps.append(p); rows.append((c, b, cc))
    tests['corr_all'] = {c: {'p': p, 'p_holm': a, 'os_only': b, 'base_only': cc} for (c, b, cc), p, a in zip(rows, ps, holm(ps))}
    # ablations vs full (clean, missing, corruption, yaw)
    abt = {}
    for c in ABL[1:]:
        if c not in acc:
            continue
        abt[c] = {}
        for grp, ks in {'clean': ['clean'], 'missing': ['avail_' + p for p in PATS],
                        'corr': [f'cor_{m}_{k}_{s}' for m, k in CORR for s in (1, 3, 5)], 'yaw': ['yaw_30', 'yaw_60']}.items():
            c1 = np.concatenate([corr_vec[('opensense', k)] for k in ks]); c2 = np.concatenate([corr_vec[(c, k)] for k in ks])
            p, b, cc = mcnemar(c1, c2)
            abt[c][grp] = {'p': p, 'd': float(c1.mean() - c2.mean())}
    tests['ablation'] = abt
    S['tests'] = tests
    # Friedman / Nemenyi over the 39 conditions for fusion models
    fm = [c for c in FUSION if c in acc]
    M = np.array([[acc[c][k] for c in fm] for k in keys])
    ranks = np.array([stats.rankdata(-row) for row in M])
    fr = stats.friedmanchisquare(*[M[:, j] for j in range(len(fm))])
    k_, N_ = len(fm), len(keys)
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}[k_]
    cd = q * np.sqrt(k_ * (k_ + 1) / (6 * N_))
    S['friedman'] = {'models': fm, 'mean_rank': ranks.mean(0).tolist(), 'chi2': float(fr.statistic), 'p': float(fr.pvalue), 'cd': float(cd), 'N': N_}
    # reliability weights vs severity (opensense)
    rel = {}
    os_runs = runs['opensense']
    for m, kind in [('u', 'occlude'), ('s', 'occlude'), ('v', 'occlude'), ('u', 'noise'), ('s', 'noise'), ('v', 'noise')]:
        rel[f'{m}_{kind}'] = {}
        for s in (0, 1, 3, 5):
            key = 'clean' if s == 0 else f'cor_{m}_{kind}_{s}'
            W = np.concatenate([r[key]['w'] for r in os_runs])
            rel[f'{m}_{kind}'][s] = W.mean(0).tolist()
    S['reliability'] = rel
    return S


# ------------------------------------------------------------------ open world

def open_summary():
    S = {}
    splits = ['mirror', 'shuffle', 'holistic']
    for sp in splits:
        S[sp] = {}
        for cfg in ['opensense', 'langbind', 'abl_labeltext', 'xmtf', 'edl']:
            runs = []
            for f in range(5):
                p = f'{R}/open/{cfg}_f{f}_s0_{sp}.pkl'
                if os.path.exists(p):
                    runs.append(pickle.load(open(p, 'rb')))
            if len(runs) < 5:
                continue
            S[sp][cfg] = open_metrics(cfg, runs)
    return S


def auroc_fpr(score_unk, score_known):
    """score: higher = more likely unknown."""
    y = np.r_[np.ones(len(score_unk)), np.zeros(len(score_known))]
    s = np.r_[score_unk, score_known]
    auc = roc_auc_score(y, s)
    thr = np.percentile(score_unk, 5)            # 95% of unknowns flagged
    fpr = float((score_known >= thr).mean())
    return float(auc), fpr


def open_metrics(cfg, runs):
    out = {}
    seen, desc, unk = runs[0]['seen'], runs[0]['desc'], runs[0]['unk']
    agg = {k: [] for k in ['E_seen', 'E_desc', 'E_unk', 'msp_seen', 'msp_unk', 'maha_seen', 'maha_unk', 'zsl', 'ow_s', 'ow_d', 'ow_o',
                           'seen_acc', 'gz_s', 'gz_u', 'roc', 'E_desc_unk']}
    for r in runs:
        yte = r['y_te']
        is_s = np.isin(yte, seen); is_d = np.isin(yte, desc); is_o = np.isin(yte, unk)
        if cfg in ('xmtf', 'edl'):
            lg_te = r['out_te']['logits']; lg_va = r['out_va']['logits']
            if cfg == 'edl':
                ev = lg_te; K = ev.shape[1]
                unc = K / (ev + 1).sum(1)
                agg['E_seen'].append(unc[is_s]); agg['E_unk'].append(unc[is_o]); agg['E_desc'].append(unc[is_d])
                agg['seen_acc'].append((ev.argmax(1)[is_s] == np.searchsorted(seen, yte[is_s])).mean())
                continue
            E = -np.log(np.exp(lg_te - lg_te.max(1, keepdims=True)).sum(1)) - lg_te.max(1)
            msp = -softmax(lg_te).max(1)
            ef_tr = r['out_tr']['ef'].astype(np.float64); ef_te = r['out_te']['ef'].astype(np.float64)
            ytr = r['y_tr']
            mus = np.stack([ef_tr[ytr == c].mean(0) for c in seen])
            Xc = np.concatenate([ef_tr[ytr == c] - mus[i] for i, c in enumerate(seen)])
            cov = np.cov(Xc.T) + 1e-3 * np.eye(Xc.shape[1]); icov = np.linalg.inv(cov)
            dm = np.stack([np.einsum('nd,de,ne->n', ef_te - m, icov, ef_te - m) for m in mus], 1).min(1)
            agg['E_seen'].append(E[is_s]); agg['E_unk'].append(E[is_o]); agg['E_desc'].append(E[is_d])
            agg['msp_seen'].append(msp[is_s]); agg['msp_unk'].append(msp[is_o])
            agg['maha_seen'].append(dm[is_s]); agg['maha_unk'].append(dm[is_o])
            agg['seen_acc'].append((lg_te.argmax(1)[is_s] == np.searchsorted(seen, yte[is_s])).mean())
            continue
        # language-grounded models: three-way rule of Eq. (decide)
        P_all = r['P_all']; tau = r['tau']
        ytr, yva = r['y_tr'], r['y_va']
        nz = lambda a: a.astype(np.float64) / np.linalg.norm(a.astype(np.float64), axis=1, keepdims=True)
        if r.get('dual'):
            lg_va = r['out_va']['logits'].astype(np.float64); lg_te = r['out_te']['logits'].astype(np.float64)
            es_va = nz(r['out_va']['es']); es_te = nz(r['out_te']['es'])
        else:
            ef_va = nz(r['out_va']['ef']); ef_te = nz(r['out_te']['ef'])
            lg_va = ef_va @ P_all[seen].T / tau; lg_te = ef_te @ P_all[seen].T / tau
            es_va, es_te = ef_va, ef_te
        en = lambda lg: -np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) - lg.max(1)
        Eva, Ete = en(lg_va), en(lg_te)
        dE = np.quantile(Eva, 0.95)
        dR = np.quantile((es_va * P_all[yva]).sum(1), 0.05)
        Sd = es_te @ P_all[desc].T
        pred = np.full(len(yte), -1)
        known = Ete <= dE
        pred[known] = seen[lg_te[known].argmax(1)]
        ret = (~known) & (Sd.max(1) >= dR)
        pred[ret] = desc[Sd[ret].argmax(1)]
        agg['ow_s'].append((pred[is_s] == yte[is_s]).mean()); agg['ow_d'].append((pred[is_d] == yte[is_d]).mean())
        agg['ow_o'].append((pred[is_o] == -1).mean())
        agg['zsl'].append((desc[Sd[is_d].argmax(1)] == yte[is_d]).mean())
        agg['seen_acc'].append((seen[lg_te[is_s].argmax(1)] == yte[is_s]).mean())
        # generalised ZSL in the semantic space with calibrated stacking (gamma from seen validation data)
        Ss_va = es_va @ P_all[seen].T; Sd_va = es_va @ P_all[desc].T
        gam = max(np.quantile(Ss_va.max(1) - Sd_va.max(1), 0.05), 0.0)
        Sall = np.concatenate([es_te @ P_all[seen].T - gam, Sd], 1)
        gp = np.concatenate([seen, desc])[Sall.argmax(1)]
        agg['gz_s'].append((gp[is_s] == yte[is_s]).mean()); agg['gz_u'].append((gp[is_d] == yte[is_d]).mean())
        agg['E_seen'].append(Ete[is_s]); agg['E_unk'].append(Ete[is_o]); agg['E_desc'].append(Ete[is_d])
        msp = -softmax(lg_te).max(1)
        agg['msp_seen'].append(msp[is_s]); agg['msp_unk'].append(msp[is_o])
        # operating curve of the retrieval floor: described-unseen accuracy vs undescribed rejection
        grid = np.linspace(-0.2, 1.0, 61)
        cd_, co_ = [], []
        for th in grid:
            pr = np.full(len(yte), -1); pr[known] = seen[lg_te[known].argmax(1)]
            rr = (~known) & (Sd.max(1) >= th); pr[rr] = desc[Sd[rr].argmax(1)]
            cd_.append((pr[is_d] == yte[is_d]).mean()); co_.append((pr[is_o] == -1).mean())
        agg.setdefault('curve_d', []).append(cd_); agg.setdefault('curve_o', []).append(co_)
        agg.setdefault('dR', []).append(dR)
        agg.setdefault('nov_d', []).append((Ete[is_d] > dE).mean()); agg.setdefault('nov_o', []).append((Ete[is_o] > dE).mean())
    cat = lambda k: np.concatenate(agg[k]) if agg[k] else None
    out['seen_acc'] = float(np.mean(np.concatenate([[x] for x in agg['seen_acc']])))
    auc, fpr = auroc_fpr(cat('E_unk'), cat('E_seen')); out['auroc_energy'] = auc; out['fpr95_energy'] = fpr
    out['auroc_novel_energy'] = float(auroc_fpr(np.r_[cat('E_unk'), cat('E_desc')], cat('E_seen'))[0])
    if agg['msp_seen']:
        auc, fpr = auroc_fpr(cat('msp_unk'), cat('msp_seen')); out['auroc_msp'] = auc; out['fpr95_msp'] = fpr
    if agg['maha_seen']:
        auc, fpr = auroc_fpr(cat('maha_unk'), cat('maha_seen')); out['auroc_maha'] = auc; out['fpr95_maha'] = fpr
    if agg['zsl']:
        n_d = [np.isin(r['y_te'], desc).sum() for r in runs]
        out['zsl'] = float(np.average(agg['zsl'], weights=n_d))
        out['ow_seen'] = float(np.mean(agg['ow_s'])); out['ow_desc'] = float(np.mean(agg['ow_d']))
        out['ow_unk'] = float(np.mean(agg['ow_o']))
        out['A_ow'] = float(np.mean([out['ow_seen'], out['ow_desc'], out['ow_unk']]))
        gs, gu = float(np.mean(agg['gz_s'])), float(np.mean(agg['gz_u']))
        out['gzsl_s'], out['gzsl_u'] = gs, gu
        out['gzsl_H'] = 2 * gs * gu / (gs + gu + 1e-9)
    if agg.get('curve_d'):
        out['curve'] = [np.mean(agg['curve_d'], 0).tolist(), np.mean(agg['curve_o'], 0).tolist()]
        out['dR'] = float(np.mean(agg['dR']))
        out['nov_d'] = float(np.mean(agg['nov_d'])); out['nov_o'] = float(np.mean(agg['nov_o']))
    # ROC curve points for plotting
    y = np.r_[np.ones(len(cat('E_unk'))), np.zeros(len(cat('E_seen')))]
    fp, tp, _ = roc_curve(y, np.r_[cat('E_unk'), cat('E_seen')])
    out['roc'] = [fp[::max(1, len(fp) // 60)].tolist(), tp[::max(1, len(fp) // 60)].tolist()]
    return out


def limited_summary():
    S = {}
    for cfg in ['opensense', 'abl_nopre']:
        S[cfg] = {}
        for fr in ['0.1', '0.25']:
            accs = []
            for f in range(5):
                p = f'{R}/limited/{cfg}_f{f}_s0_{fr}.pkl'
                if os.path.exists(p):
                    r = pickle.load(open(p, 'rb'))
                    accs.append((r['clean']['logits'].argmax(1) == r['y']))
            if len(accs) == 5:
                c = np.concatenate(accs); S[cfg][fr] = acc_ci(c)
    return S


def tta_summary():
    S = {}
    for mode in ('closed', 'open'):
        fs = sorted(glob.glob(f'{R}/tta/tta_{mode}_f*.pkl'))
        if len(fs) < 5:
            continue
        rs = [pickle.load(open(f, 'rb')) for f in fs]
        methods = [k for k in rs[0]['results'] if not k.endswith('_unk_used') and not k.endswith('_n_sel')]
        S[mode] = {'domains': rs[0]['domains']}
        for m in methods:
            A = np.array([r['results'][m] for r in rs])
            S[mode][m] = {'mean': A.mean(0).tolist(), 'sd': A.std(0).tolist(),
                          'unk_used': np.sum([r['results'][m + '_unk_used'] for r in rs], 0).tolist(),
                          'n_sel': np.sum([r['results'][m + '_n_sel'] for r in rs], 0).tolist()}
    return S


if __name__ == '__main__':
    out = {}
    try:
        out['closed'] = closed_summary()
    except Exception as e:
        print('closed incomplete:', repr(e))
    out['open'] = open_summary()
    out['limited'] = limited_summary()
    out['tta'] = tta_summary()
    json.dump(out, open(f'{R}/summary.json', 'w'), indent=1, default=float)
    print('written')
