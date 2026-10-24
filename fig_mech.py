"""Fig. 2 (reliability-biased attention with substitution tokens) and Fig. 3 (open-world decision + adaptation)."""
from diagram import Box, arrow, poly_arrow, canvas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

# ------------------------------------------------------------------ Fig. 2
fig, ax = canvas(3.45, 2.9, 34.5, 29)
G1, G2 = '0.94', '0.86'
A = Box(ax, 0.5, 20.5, 8.0, 5.0, 'observed\nlatent $\\mathbf{z}_m$\n($a^m$)', fs=5.8, fill=G1)
Bx = Box(ax, 0.5, 8.5, 8.0, 5.0, 'other streams\n$\\mathbf{z}_{\\backslash m}$,\n$\\mathbf{a}_{\\backslash m}$', fs=5.8, fill=G1)
C = Box(ax, 10.5, 20.5, 8.5, 5.0, 'aleatoric head\n$s_m=h_m(\\mathbf{z}_m)$', fs=5.8)
Dg = Box(ax, 10.5, 8.5, 8.5, 5.0, 'Gaussian\ngenerator\n$\\boldsymbol{\\mu}_\\phi,\\ \\boldsymbol{\\ell}_\\phi$', fs=5.8)
E = Box(ax, 10.5, 14.75, 8.5, 4.0, 'gate $g_m$', fs=5.8, fill=G2)
F = Box(ax, 21.0, 8.5, 5.5, 17.0, 'mix\n$\\tilde{\\mathbf{z}}_m$\n\nEq. (14)', fs=5.8, fill=G2)
Gp = Box(ax, 28.0, 15.5, 6.0, 10.0, '$\\pi_m$ +\nMC\ndropout\n\n$\\mathbf{e}_m$,\n$\\sigma^2_m$', fs=5.8)
H = Box(ax, 28.0, 1.0, 6.0, 9.5, 'PoE\nfusion\n$\\Sigma_k\\omega_k\\mathbf{e}_k$', fs=5.8, fill='0.80')
O = Box(ax, 14.0, 1.0, 12.5, 5.0, 'other experts $\\mathbf{e}_k,\\ \\sigma^2_k$\n($k\\neq m$, incl. joint)', fs=5.6, fill=G1)
arrow(ax, A.r(0.5), C.l(0.5))
arrow(ax, Bx.r(0.5), Dg.l(0.5))
arrow(ax, C.b(0.5), E.t(0.5))
arrow(ax, E.r(0.5), (21.0, 16.75))
arrow(ax, Dg.r(0.5), (21.0, 11.0))
poly_arrow(ax, [A.t(0.5), (4.5, 27.5), (23.75, 27.5), (23.75, 25.5)])
arrow(ax, (26.5, 20.5), (28.0, 20.5))
arrow(ax, Gp.b(0.5), H.t(0.5))
arrow(ax, O.r(0.5), (28.0, 3.5))
ax.text(14.2, 27.9, 'own latent', fontsize=5.4, ha='center', va='bottom', style='italic')
fig.savefig('fig2_rba.png', dpi=300)

# ------------------------------------------------------------------ Fig. 3
fig, ax = canvas(3.45, 3.3, 34.5, 33)
IN = Box(ax, 9.5, 28.5, 15.5, 3.6, 'Fused embedding $\\mathbf{e}$, energy $E(\\mathbf{x})$', fs=6.2, fill='0.94')


def diamond(cx, cy, w, h, text):
    ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)], closed=True,
                         fc='white', ec='black', lw=0.8, zorder=2))
    ax.text(cx, cy, text, fontsize=6.0, ha='center', va='center', zorder=3)


diamond(17.25, 22.3, 13.0, 6.2, '$E(\\mathbf{x})\\leq\\delta_E$ ?')
KN = Box(ax, 0.5, 13.0, 11.5, 4.6, 'Known activity\n$\\hat{y}=\\arg\\max_{c\\in\\mathcal{C}_s}\\;\\mathbf{e}^{\\top}\\mathbf{w}_c$', fs=6.0, fill='0.88')
diamond(25.0, 15.3, 13.0, 6.2, '$\\max_{c\\in\\mathcal{C}_d}\\;\\mathbf{e}^{\\mathrm{sem}\\top}\\mathbf{t}_c\\geq\\delta_R$ ?')
UN = Box(ax, 18.5, 5.3, 8.6, 4.4, 'Described\nunseen activity', fs=6.0, fill='0.88')
RJ = Box(ax, 27.8, 5.3, 5.0, 4.4, 'Reject', fs=6.2, fill='0.94')
FL = Box(ax, 0.5, 5.3, 11.5, 4.4, '$\\mathcal{H}(\\mathbf{q})\\leq h_0$ ?\nadd to subset $\\mathcal{S}$', fs=6.0)
AD = Box(ax, 0.5, 0.3, 26.6, 3.4, 'Update adapters, Eq. (20), and drift-bounded\nprototype memory $\\boldsymbol{\\mu}_c$, Eq. (21)', fs=5.9, ls='--')
arrow(ax, IN.b(), (17.25, 25.4))
poly_arrow(ax, [(10.75, 22.3), (6.25, 22.3), (6.25, 17.6)])
ax.text(8.5, 22.75, 'yes', fontsize=5.8, ha='center', va='bottom')
poly_arrow(ax, [(23.75, 22.3), (25.0, 22.3), (25.0, 18.4)])
ax.text(24.4, 22.75, 'no', fontsize=5.8, ha='center', va='bottom')
poly_arrow(ax, [(18.5, 15.3), (16.0, 15.3), (16.0, 11.5), (22.8, 11.5), (22.8, 9.7)])
ax.text(17.3, 15.75, 'yes', fontsize=5.8, ha='center', va='bottom')
poly_arrow(ax, [(31.5, 15.3), (32.3, 15.3), (32.3, 9.7)])
ax.text(32.0, 15.75, 'no', fontsize=5.8, ha='center', va='bottom')
arrow(ax, KN.b(), FL.t())
arrow(ax, FL.b(0.5), (6.25, 3.7), ls='--')
# dashed feedback from the update to the model (reverse path)
poly_arrow(ax, [(27.1, 2.0), (33.9, 2.0), (33.9, 30.3), (25.0, 30.3)], ls='--')
ax.text(33.5, 25.0, 'adapted\nmodel', fontsize=5.4, ha='right', va='center')
fig.savefig('fig3_decision.png', dpi=300)
print('ok')
