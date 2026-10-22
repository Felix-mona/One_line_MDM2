"""
Plotting.py - helpers for exploring "what happens if I change X?"

Quick start (in Main.py):

    import Plotting as P
    image = Image("Example_images/smiley.svg", 200); image.sample_paths()
    pts = image.points

    P.plot_spectrum(pts)                         # where is the energy?
    P.plot_reconstructions(pts, [1, 3, 5, 10, 20, 50, 100])   # small multiples
    res = P.sweep(pts, alpha_beta_pairs=[(1, 0), (0.5, 0.5), (0.3, 0.7)])
    P.plot_metrics(res)                          # accuracy / smoothness / loss vs k
    P.plot_best_overlay(pts, res["best"])        # best k for each alpha/beta
    slider = P.interactive_explorer(pts)         # drag k and watch the curve

Everything takes `pts`, a 1D complex array of the ORIGINAL sampled points.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import Fourier_reconsctruction as fourier
import Metric as met


# ----------------------------------------------------------------------------
# core helpers
# ----------------------------------------------------------------------------
def reconstruct(points, k, dense=None):
    """
    Low-pass reconstruction keeping frequencies -k..k.
    dense=None  -> same number of points as the input (what Metric uses)
    dense=int   -> evaluate the Fourier series at that many points (smooth plot)
    """
    coeffs, freqs = fourier.fourier_coeffs(points)
    filtered = fourier.filter_coeffs(coeffs, k)
    if dense is None:
        return fourier.inverse_fourier_descriptors(filtered)

    t = np.linspace(0, 1, dense, endpoint=False)
    keep = filtered != 0
    basis = np.exp(2j * np.pi * np.outer(t, freqs[keep]))
    return basis @ filtered[keep]


def default_ks(points, num=12):
    """Roughly geometric, integer, unique cutoffs from 1 to N/2."""
    n = len(points)
    return np.unique(np.geomspace(1, n // 2, num).astype(int))


def _close(z):
    return np.append(z, z[0])


def _style(ax, title=""):
    ax.set_aspect("equal")
    ax.invert_yaxis()  # SVG y axis points down
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


# ----------------------------------------------------------------------------
# 1. spectrum
# ----------------------------------------------------------------------------
def plot_spectrum(points, save=None):
    """Coefficient magnitudes and cumulative energy captured vs cutoff k."""
    coeffs, freqs = fourier.fourier_coeffs(points)
    order = np.argsort(freqs)
    n = len(points)

    ks = np.arange(0, n // 2 + 1)
    power = np.abs(coeffs) ** 2
    total = power.sum()
    captured = np.array([power[np.abs(freqs) <= k].sum() / total for k in ks])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.semilogy(freqs[order], np.abs(coeffs[order]) + 1e-16)
    a1.set(title="|coefficient| by frequency", xlabel="frequency", ylabel="magnitude")
    a1.grid(alpha=0.3)

    a2.plot(ks, captured)
    for level in (0.9, 0.99):
        k_at = ks[np.argmax(captured >= level)]
        a2.axhline(level, ls=":", c="grey")
        a2.annotate(f"{level:.0%} at k={k_at}", (k_at, level),
                    textcoords="offset points", xytext=(8, -14))
    a2.set(title="Energy captured by cutoff k", xlabel="cutoff k",
           ylabel="fraction of energy")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    plt.show()


# ----------------------------------------------------------------------------
# 2. small multiples
# ----------------------------------------------------------------------------
def plot_reconstructions(points, ks=None, dense=800, ncols=4,
                         show_samples=False, save=None):
    """Grid: original (grey) vs reconstruction (blue) for each cutoff."""
    ks = default_ks(points) if ks is None else list(ks)
    nrows = int(np.ceil(len(ks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
                             squeeze=False)
    orig = _close(points)

    for ax, k in zip(axes.flat, ks):
        rec = reconstruct(points, k, dense=dense)
        acc = met.calc_metric(reconstruct(points, k), points, "accuracy")
        ax.plot(orig.real, orig.imag, color="0.75", lw=2)
        ax.plot(_close(rec).real, _close(rec).imag, color="tab:blue", lw=1.5)
        if show_samples:
            ax.plot(points.real, points.imag, ".", ms=2, color="tab:red")
        _style(ax, f"k={k}   Hausdorff={acc:.3f}")
    for ax in list(axes.flat)[len(ks):]:
        ax.axis("off")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    plt.show()


# ----------------------------------------------------------------------------
# 3. metric sweep (replacement for gather_lowest_metric)
# ----------------------------------------------------------------------------
def sweep(points, ks=None, alpha_beta_pairs=((0.5, 0.5),), smoothness_fn=None):
    """
    Evaluate accuracy, smoothness and loss for every k.

    smoothness_fn(new_points, old_points) -> float in [0,1], lower = better.
    Defaults to met.calc_metric(..., 'smoothness') (currently a 0.5 stub).

    Returns dict with keys: ks, accuracy, smoothness, loss {(a,b): array},
    best {(a,b): k}
    """
    ks = default_ks(points) if ks is None else np.asarray(ks)
    if smoothness_fn is None:
        smoothness_fn = lambda new, old: met.calc_metric(new, old, "smoothness")

    acc = np.empty(len(ks))
    smo = np.empty(len(ks))
    for i, k in enumerate(ks):
        new = reconstruct(points, k)
        acc[i] = met.calc_metric(new, points, "accuracy")
        smo[i] = smoothness_fn(new, points)

    loss = {(a, b): a * acc + b * smo for a, b in alpha_beta_pairs}
    best = {ab: int(ks[np.argmin(l)]) for ab, l in loss.items()}
    return {"ks": ks, "accuracy": acc, "smoothness": smo, "loss": loss, "best": best}


def plot_metrics(result, save=None):
    """Accuracy/smoothness vs k, and loss vs k for each alpha/beta with the best k marked."""
    ks = result["ks"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))

    a1.plot(ks, result["accuracy"], "o-", label="accuracy (Hausdorff, lower=better)")
    a1.plot(ks, result["smoothness"], "s-", label="smoothness penalty")
    a1.set(title="Individual metrics", xlabel="cutoff k", ylim=(0, 1))
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)

    for (a, b), l in result["loss"].items():
        line, = a2.plot(ks, l, "o-", ms=3, label=f"alpha={a}, beta={b}")
        kb = result["best"][(a, b)]
        a2.plot(kb, l[list(ks).index(kb)], "*", ms=14, color=line.get_color())
    a2.set(title="Loss (star = best k)", xlabel="cutoff k", ylabel="loss")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    plt.show()


# ----------------------------------------------------------------------------
# 4. overlay of best results
# ----------------------------------------------------------------------------
def plot_best_overlay(points, best, dense=800, save=None):
    """best: {(alpha,beta): k}. Overlays original + one reconstruction per pair."""
    fig, ax = plt.subplots(figsize=(8, 8))
    orig = _close(points)
    ax.plot(orig.real, orig.imag, color="0.7", lw=4, label="original")

    for (a, b), k in best.items():
        rec = _close(reconstruct(points, k, dense=dense))
        ax.plot(rec.real, rec.imag, lw=1.5, label=f"a={a}, b={b} -> k={k}")
    _style(ax, "Best reconstruction per alpha/beta")
    ax.legend(fontsize=8)
    if save:
        fig.savefig(save, dpi=150)
    plt.show()


# ----------------------------------------------------------------------------
# 5. interactive slider
# ----------------------------------------------------------------------------
def interactive_explorer(points, dense=1000):
    """
    Drag the slider to change the cutoff k. Keep the returned slider alive
    (assign it to a variable) or matplotlib will garbage-collect it.
    Needs an interactive backend (a normal desktop window, not a static one).
    """
    n = len(points)
    fig, ax = plt.subplots(figsize=(8, 8.8))
    plt.subplots_adjust(bottom=0.12)

    orig = _close(points)
    ax.plot(orig.real, orig.imag, color="0.75", lw=4)
    k0 = max(1, n // 20)
    rec = _close(reconstruct(points, k0, dense=dense))
    line, = ax.plot(rec.real, rec.imag, color="tab:blue", lw=1.5)
    _style(ax)

    def title(k):
        acc = met.calc_metric(reconstruct(points, k), points, "accuracy")
        ax.set_title(f"k={k}   Hausdorff={acc:.4f}   "
                     f"coefficients kept={min(2 * k + 1, n)}/{n}")

    title(k0)
    slider = Slider(fig.add_axes([0.15, 0.04, 0.7, 0.03]), "cutoff k",
                    0, n // 2, valinit=k0, valstep=1)

    def update(val):
        k = int(slider.val)
        r = _close(reconstruct(points, k, dense=dense))
        line.set_data(r.real, r.imag)
        title(k)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    return slider
