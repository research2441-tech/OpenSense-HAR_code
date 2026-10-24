"""Result figures (grayscale, legends placed outside the data area)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 7, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False,
                     'mathtext.fontset': 'dejavuserif'})
S = json.load(open('results/summary.json'))
OUT = './'
NAMES = {'late': 'Late fusion', 'early': 'Early fusion', 'attn': 'HAMLET-style', 'xmtf': 'XM-TF',
         'actionmae': 'ActionMAE-style', 'langbind': 'Lang.-binding', 'opensense': 'OpenSense-HAR',
         'stgcn': 'ST-GCN-style', 'dcl': 'DeepConvLSTM', 'depth': 'Depth TCN'}
GRAYS = ['0.92', '0.80', '0.68', '0.55', '0.42', '0.28', '0.05']
HATCH = ['', '////', '', '\\\\\\\\', '', 'xxx', '']
MK = ['o', 's', '^', 'v', 'D', 'P', '*']
LS = [':', '--', '-.', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), '-']


def fig_missing():
    acc = S['closed']['acc']
    models = [m for m in ['late', 'early', 'attn', 'xmtf', 'actionmae', 'langbind', 'opensense'] if m in acc]
    pats = ['clean', 'avail_vs', 'avail_vu', 'avail_su', 'avail_v', 'avail_s', 'avail_u']
    labels = ['all', 'D+S', 'D+I', 'S+I', 'D only', 'S only', 'I only']
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    w = 0.8 / len(models)
    x = np.arange(len(pats))
    for i, m in enumerate(models):
        v = [100 * acc[m][p] for p in pats]
        ax.bar(x - 0.4 + w * (i + 0.5), v, w, color=GRAYS[i], edgecolor='black', linewidth=0.4, hatch=HATCH[i],
               label=NAMES[m])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(40, 100)
    ax.grid(axis='y', lw=0.3, color='0.8'); ax.set_axisbelow(True)
    ax.legend(ncol=7, loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=6.4, handlelength=1.6, columnspacing=1.0)
    ax.set_xlabel('Available streams at test time (D = depth, S = skeleton, I = inertial)')
    fig.tight_layout()
    fig.savefig(OUT + 'fig4_missing.png', dpi=300)


def fig_corruption():
    acc = S['closed']['acc']
    models = [m for m in ['early', 'attn', 'xmtf', 'actionmae', 'langbind', 'opensense'] if m in acc]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=False)
    titles = {'v': 'Depth corruptions', 's': 'Skeleton corruptions', 'u': 'Inertial corruptions'}
    for j, mod in enumerate('vsu'):
        ax = axs[j]
        for i, m in enumerate(models):
            y = [100 * acc[m]['clean']] + [100 * acc[m][f'cor_{mod}_sev{s}'] for s in (1, 3, 5)]
            ax.plot([0, 1, 3, 5], y, color='black' if m == 'opensense' else '0.45', ls=LS[i + 1], marker=MK[i + 1],
                    ms=3.2, lw=1.3 if m == 'opensense' else 0.8, label=NAMES[m], mfc='white' if m != 'opensense' else 'black')
        ax.set_title(titles[mod], fontsize=7)
        ax.set_xticks([0, 1, 3, 5]); ax.set_xlabel('Severity (0 = clean)')
        ax.grid(lw=0.3, color='0.85')
        if j == 0:
            ax.set_ylabel('Mean accuracy (%)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.0), fontsize=6.4, handlelength=2.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT + 'fig5_corruption.png', dpi=300)


def fig_reliability():
    rel = S['closed']['reliability']
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.15))
    cases = [('u_occlude', 'Dead IMU units'), ('s_occlude', 'Skeleton limb loss'), ('v_occlude', 'Depth occlusion')]
    for j, (key, title) in enumerate(cases):
        ax = axs[j]
        sev = [0, 1, 3, 5]
        W = np.array([rel[key][str(s)] for s in sev])     # columns: w_v w_s w_u w_j
        for i, (lab, ls, mk) in enumerate([('$\\omega_v$ depth', '-', 'o'), ('$\\omega_s$ skeleton', '--', 's'),
                                           ('$\\omega_u$ inertial', ':', '^'), ('$\\omega_j$ joint', '-.', 'D')]):
            ax.plot(sev, W[:, i], color='black' if i < 3 else '0.5', ls=ls, marker=mk, ms=3.2, lw=1.0, label=lab, mfc='white')
        ax.set_title(title, fontsize=7); ax.set_xticks(sev); ax.set_xlabel('Severity (0 = clean)')
        ax.set_ylim(0.15, 0.32); ax.grid(lw=0.3, color='0.85')
        if j == 0:
            ax.set_ylabel('Mean fusion weight')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.0), fontsize=6.4)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT + 'fig6_reliability.png', dpi=300)


def fig_open():
    O = S['open']
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for j, sp in enumerate(['mirror', 'shuffle', 'holistic']):
        ax = axs[j]
        for i, (m, lab) in enumerate([('opensense', 'OpenSense-HAR (energy)'), ('langbind', 'Lang.-binding (energy)'),
                                      ('xmtf', 'XM-TF (energy)'), ('edl', 'Evidential')]):
            if m in O.get(sp, {}):
                fp, tp = O[sp][m]['roc']
                ax.plot(fp, tp, color='black' if m == 'opensense' else '0.45', ls=LS[[6, 2, 1, 3][i]], lw=1.3 if m == 'opensense' else 0.9,
                        label=lab)
        ax.plot([0, 1], [0, 1], color='0.7', lw=0.5)
        ax.set_title(sp.capitalize() + ' split', fontsize=7); ax.set_xlabel('False-positive rate')
        if j == 0:
            ax.set_ylabel('True-positive rate (undescribed)')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.0); ax.grid(lw=0.3, color='0.85')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.0), fontsize=6.4, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT + 'fig7_open.png', dpi=300)


def fig_owcurve():
    O = S['open']
    grid = np.linspace(-0.2, 1.0, 61)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for j, sp in enumerate(['mirror', 'shuffle', 'holistic']):
        ax = axs[j]
        for i, (m, lab) in enumerate([('opensense', 'OpenSense-HAR (descriptions)'), ('abl_labeltext', 'OpenSense-HAR (label names)'),
                                      ('langbind', 'Language-binding')]):
            if m not in O.get(sp, {}) or 'curve' not in O[sp][m]:
                continue
            cd_, co_ = O[sp][m]['curve']
            ax.plot(100 * np.array(co_), 100 * np.array(cd_), color=['black', '0.35', '0.6'][i], ls=['-', '--', ':'][i], lw=1.1, label=lab)
            k = int(np.argmin(abs(grid - O[sp][m]['dR'])))
            ax.plot([100 * co_[k]], [100 * cd_[k]], marker='o', ms=4, color=['black', '0.35', '0.6'][i], mfc='white')
        ax.set_title(sp.capitalize() + ' split', fontsize=7)
        ax.set_xlabel('Undescribed activities rejected (%)')
        if j == 0:
            ax.set_ylabel('Described unseen correctly named (%)')
        ax.set_xlim(-2, 102); ax.set_ylim(-2, 102); ax.grid(lw=0.3, color='0.85')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0), fontsize=6.4, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT + 'fig10_owcurve.png', dpi=300)


def fig_tta():
    T = S['tta']
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={'width_ratios': [1.25, 1]})
    for j, mode in enumerate(['closed', 'open']):
        if mode not in T:
            continue
        ax = axs[j]
        ms = [m for m in ['source', 'tent', 'eata', 'cotta', 'ours'] if m in T[mode]]
        labs = {'source': 'Source', 'tent': 'Tent', 'eata': 'EATA', 'cotta': 'CoTTA', 'ours': 'OpenSense-TTA'}
        x = np.arange(len(T[mode]['domains']))
        for i, m in enumerate(ms):
            ax.plot(x, 100 * np.array(T[mode][m]['mean']), color='black' if m == 'ours' else '0.45', ls=LS[[0, 1, 2, 3, 6][i]],
                    marker=MK[i], ms=3.2, lw=1.3 if m == 'ours' else 0.9, mfc='white' if m != 'ours' else 'black', label=labs[m])
        ax.set_xticks(x); ax.set_xticklabels([f'D{k}' for k in x])
        ax.set_xlabel('Deployment stage'); ax.grid(lw=0.3, color='0.85')
        ax.set_title('Closed-set stream' if mode == 'closed' else 'Stream with unseen activities (seen-class accuracy)', fontsize=7)
        if j == 0:
            ax.set_ylabel('Accuracy (%)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.0), fontsize=6.4, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT + 'fig8_tta.png', dpi=300)


def fig_cd():
    F = S['closed']['friedman']
    ms = F['models']; r = np.array(F['mean_rank']); cd = F['cd']
    o = np.argsort(r)
    fig, ax = plt.subplots(figsize=(3.45, 1.65))
    lo, hi = 1, len(ms)
    ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(-0.02, 1.12)
    ax.hlines(0.78, lo, hi, color='black', lw=0.7)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0.74, 0.82, color='black', lw=0.6); ax.text(t, 0.88, str(t), ha='center', fontsize=6.5)
    for k, idx in enumerate(o):
        y = 0.62 - 0.085 * k
        left = k < len(o) / 2
        ax.vlines(r[idx], y, 0.78, color='black', lw=0.5)
        xe = lo - 0.25 if left else hi + 0.25
        ax.hlines(y, r[idx], xe, color='black', lw=0.5)
        ax.text(xe + (-0.05 if left else 0.05), y, f'{NAMES[ms[idx]]} ({r[idx]:.2f})', ha='right' if left else 'left',
                va='center', fontsize=6.0)
    ax.hlines(1.0, lo, lo + cd, color='black', lw=1.0)
    ax.text(lo + cd / 2, 1.02, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=6.0)
    # cliques: groups whose rank difference is below CD
    rs = r[o]; y0 = 0.70; drawn = []
    for a in range(len(rs)):
        b = a
        while b + 1 < len(rs) and rs[b + 1] - rs[a] < cd:
            b += 1
        if b > a and not any(a >= p and b <= q for p, q in drawn):
            ax.hlines(y0 - 0.02 * len(drawn), rs[a] - 0.03, rs[b] + 0.03, color='black', lw=2.0)
            drawn.append((a, b))
    ax.axis('off')
    fig.subplots_adjust(left=0.31, right=0.69, top=0.97, bottom=0.02)
    fig.savefig(OUT + 'fig9_cd.png', dpi=300)


if __name__ == '__main__':
    for f in (fig_missing, fig_corruption, fig_reliability, fig_open, fig_owcurve, fig_tta, fig_cd):
        try:
            f(); print('ok', f.__name__)
        except Exception as e:
            print('skip', f.__name__, repr(e))
