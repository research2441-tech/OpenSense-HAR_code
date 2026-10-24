"""Fig. 1 - OpenSense-HAR overall architecture (grayscale)."""
from diagram import Box, arrow, poly_arrow, canvas
import matplotlib.pyplot as plt

fig, ax = canvas(7.2, 4.4, 72, 40)
ax.set_ylim(-4, 40)
G1, G2, G3 = '#f2f2f2', '#e0e0e0', '#cfcfcf'

# ---------------- inputs and tokenisers
V = Box(ax, 0.5, 26, 9.6, 5.5, 'Depth video\n$V\\in\\mathbb{R}^{16\\times32\\times32}$', fill=G1)
S = Box(ax, 0.5, 18, 9.6, 5.5, '3-D skeleton\n$S\\in\\mathbb{R}^{64\\times25\\times3}$', fill=G1)
U = Box(ax, 0.5, 10, 9.6, 5.5, 'Body-worn IMUs\n$U\\in\\mathbb{R}^{128\\times60}$', fill=G1)
TV = Box(ax, 11.6, 26, 8.9, 5.5, 'Frame CNN\n$\\to$ 16 tokens')
TS = Box(ax, 11.6, 18, 8.9, 5.5, 'Pose patches\n(4 frames) $\\to$ 16')
TU = Box(ax, 11.6, 10, 8.9, 5.5, 'IMU patches\n(8 samples) $\\to$ 16')
for a, b in ((V, TV), (S, TS), (U, TU)):
    arrow(ax, a.r(), b.l())

# ---------------- shared hierarchical encoder (container)
E = Box(ax, 22, 9.5, 11, 22.5, '', fill='white', lw=1.0)
ax.text(27.5, 30.2, 'Shared hierarchical\ntemporal encoder', ha='center', va='center', fontsize=6.6, fontweight='bold', zorder=3)
L1 = Box(ax, 22.7, 21.5, 9.6, 6, 'L1: local attention\n(4-token windows,\nmotion primitives)', fill=G1, fs=5.6)
L2 = Box(ax, 22.7, 11.5, 9.6, 6.5, 'Merge pairs; L2:\nglobal attention\n(long horizon)', fill=G1, fs=5.6)
arrow(ax, L1.b(), L2.t())
for b, y in ((TV, 28.75), (TS, 20.75), (TU, 12.75)):
    arrow(ax, b.r(), (22, y))

# ---------------- latents
Z = Box(ax, 35, 9.5, 7.5, 22.5, '', fill='white', lw=1.0)
ax.text(38.75, 30.6, 'Latents', ha='center', va='center', fontsize=6.6, fontweight='bold', zorder=3)
Box(ax, 35.8, 21.5, 5.9, 7.5, '$\\mathbf{z}_v$\n$\\mathbf{z}_s$\n$\\mathbf{z}_u$\nper\nmodality', fill=G1, fs=5.8)
Box(ax, 35.8, 11.0, 5.9, 7.0, 'Joint\nexpert\n$\\mathbf{z}_j$', fill=G1, fs=5.8)
arrow(ax, (33, 20.0), (35, 20.0))

# ---------------- reliability and generator
R = Box(ax, 44.5, 21.5, 10.5, 7.5, 'Reliability heads\naleatoric $\\sigma^2_k$\nMC-dropout spread $\\upsilon_k$', fill=G2, fs=6.2)
GN = Box(ax, 44.5, 10.0, 10.5, 8.5, 'Uncertainty-gated\nlatent substitution\n$p_\\phi(\\mathbf{z}_m\\mid\\mathbf{z}_{\\backslash m})$\n(missing / unreliable)', fill=G2, fs=6.2)
arrow(ax, (42.5, 25.25), R.l())
arrow(ax, (42.5, 14.25), GN.l())
arrow(ax, R.b(0.5), GN.t(0.5))
ax.text(50.4, 20.0, 'gate $g_m$', fontsize=5.8, ha='left', va='center', zorder=5)

# ---------------- fusion and decision
F = Box(ax, 57, 11, 8.3, 18, 'Product of\nexperts\nfusion\n\n$\\mathbf{e}\\propto\\Sigma_k\\,\\omega_k\\mathbf{e}_k$\n$\\omega_k\\propto\\tilde{a}_k/\\sigma^2_k$\n\nclosed-set:\nprototypes $\\mathbf{w}_c$', fill=G3, fs=5.8)
arrow(ax, R.r(0.5), (57, 25.25))
arrow(ax, GN.r(0.5), (57, 14.25))
Dd = Box(ax, 66.5, 11, 5.2, 18, 'Open-\nworld\ndecision\n\nknown\n\nunseen\n(retrieved)\n\nreject', fill=G1, fs=5.6)
arrow(ax, F.r(), Dd.l())

# ---------------- language branch (top lane)
TX = Box(ax, 0.5, 33.5, 9.6, 5.5, 'Activity\ndescriptions $d_{c,j}$', fill=G1, fs=6.0)
TE = Box(ax, 11.6, 33.5, 8.9, 5.5, 'GloVe tokens +\nattention pooling', fill='white')
PR = Box(ax, 57, 32.75, 8.3, 7, 'Semantic head\n$\\mathbf{e}^{\\mathrm{sem}}$ vs. text\nprototypes $\\mathbf{t}_c$', fill=G2, fs=5.6)
arrow(ax, TX.r(), TE.l())
arrow(ax, TE.r(), PR.l(0.5))
ax.text(38.5, 36.7, 'text prototypes (sensor-language alignment)', fontsize=5.8, ha='center', va='bottom', style='italic')
poly_arrow(ax, [PR.r(0.5), (69.1, 36.25), (69.1, 29.0)])
arrow(ax, (61.15, 29.0), (61.15, 32.75))

# ---------------- training-only branch (dashed)
DE = Box(ax, 11.6, -3.0, 8.9, 6, 'Masked\nmultimodal decoder\n(training only)', ls='--', fs=5.6)
OB = Box(ax, 22, -3.0, 11, 6, 'Training objective\nstage 1: $\\mathcal{L}_{\\mathrm{mask}}$\nstage 2: $\\mathcal{L}_{\\mathrm{cls}},\\mathcal{L}_{\\mathrm{s2t}},\\mathcal{L}_{\\mathrm{t2s}},$\n$\\mathcal{L}_{\\mathrm{rel}},\\mathcal{L}_{\\mathrm{gen}}$', ls='--', fs=5.5)
poly_arrow(ax, [(23.2, 9.5), (23.2, 6.0), (16.05, 6.0), (16.05, 3.0)], ls='--')
arrow(ax, DE.r(0.5), OB.l(0.5), ls='--')
poly_arrow(ax, [(61.15, 11.0), (61.15, 0.0), (33.0, 0.0)], ls='--')
# reverse-mode gradient into the shared encoder
arrow(ax, (30.5, 3.0), (30.5, 9.5), ls=(0, (3, 2)), lw=1.0, head=6)
ax.text(31.1, 6.6, 'reverse-mode\ngradient $\\nabla_{\\theta}$', fontsize=5.6, ha='left', va='center', style='italic')
ax.text(47.0, 0.4, 'training signals', fontsize=5.6, ha='center', va='bottom', style='italic')

fig.savefig('fig1_arch.png', dpi=300)
print('ok')
