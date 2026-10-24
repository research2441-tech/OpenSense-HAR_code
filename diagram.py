"""Grayscale block-diagram helpers with explicit anchor points so arrows meet box edges exactly."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif'})


class Box:
    def __init__(self, ax, x, y, w, h, text, fs=6.4, fill='white', lw=0.8, ls='-', bold=False, round_=0.025,
                 va='center', tcolor='black', tpos=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={round_}', fc=fill, ec='black',
                           lw=lw, ls=ls, zorder=2)
        ax.add_patch(p)
        tx, ty = (x + w / 2, y + h / 2) if tpos is None else tpos
        ax.text(tx, ty, text, ha='center', va=va, fontsize=fs, zorder=3, color=tcolor,
                fontweight='bold' if bold else 'normal', linespacing=1.15)

    # anchors
    def l(self, f=0.5): return (self.x, self.y + f * self.h)
    def r(self, f=0.5): return (self.x + self.w, self.y + f * self.h)
    def t(self, f=0.5): return (self.x + f * self.w, self.y + self.h)
    def b(self, f=0.5): return (self.x + f * self.w, self.y)


def arrow(ax, p, q, ls='-', lw=0.8, head=5, color='black', rad=0.0, shrink=0.0, both=False):
    if ls != '-' and rad == 0.0 and not both:
        # dashed shaft with a solid head, so the head keeps a clean outline
        import math
        dx, dy = q[0] - p[0], q[1] - p[1]
        L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
        d = min(0.9, 0.4 * L)
        m = (q[0] - ux * d, q[1] - uy * d)
        ax.plot([p[0], m[0]], [p[1], m[1]], color=color, lw=lw, ls=ls, zorder=4)
        p, ls = m, '-'
    style = '<|-|>' if both else '-|>'
    a = FancyArrowPatch(p, q, arrowstyle=f'{style},head_length={head * 0.09:.2f},head_width={head * 0.055:.2f}',
                        connectionstyle=f'arc3,rad={rad}', lw=lw, ls=ls, color=color, zorder=4,
                        shrinkA=shrink, shrinkB=shrink, mutation_scale=10)
    ax.add_patch(a)


def poly_arrow(ax, pts, ls='-', lw=0.8, head=5, color='black'):
    """Orthogonal multi-segment connector; arrow head on the final segment only."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=4, solid_capstyle='butt')
    arrow(ax, pts[-2], pts[-1], ls=ls, lw=lw, head=head, color=color)


def canvas(w_in, h_in, xmax, ymax):
    fig = plt.figure(figsize=(w_in, h_in), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.axis('off')
    return fig, ax
