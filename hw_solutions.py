#!/usr/bin/env python3
"""
============================================================================
SOLUTIONS — Active Learning with Gaussian Processes
============================================================================
This file contains the completed implementations for all 14 TODOs.
Run it end-to-end to produce all figures and the animation .gif.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel as C,
    WhiteKernel,
    Matern,
)

# Reproducibility
RNG_SEED = 42
rng = np.random.RandomState(RNG_SEED)


# ═══════════════════════════════════════════════════════════════════════════
# TARGET FUNCTIONS  (identical to homework)
# ═══════════════════════════════════════════════════════════════════════════


def sigmoid_1d(x, slope=5.0, shift=0.0):
    return 1.0 / (1.0 + np.exp(-slope * (x - shift)))


def inverted_gaussian_2d(X, sigma_x=1.0, sigma_y=0.5):
    X = np.asarray(X)
    x, y = X[:, 0], X[:, 1]
    norm = 1.0 / (2.0 * np.pi * sigma_x * sigma_y)
    exponent = -0.5 * (x**2 / sigma_x**2 + y**2 / sigma_y**2)
    return -norm * np.exp(exponent)


def plot_2d_target(sigma_x=1.0, sigma_y=0.5, grid_n=80):
    xs = np.linspace(-3 * sigma_x, 3 * sigma_x, grid_n)
    ys = np.linspace(-3 * sigma_y, 3 * sigma_y, grid_n)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    Zg = inverted_gaussian_2d(pts, sigma_x, sigma_y).reshape(Xg.shape)

    fig, ax = plt.subplots(figsize=(7, 4))
    cf = ax.contourf(Xg, Yg, Zg, levels=30, cmap="viridis")
    ax.contour(Xg, Yg, Zg, levels=10, colors="k", linewidths=0.4)
    fig.colorbar(cf, ax=ax, label="f(x,y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Target: inverted Gaussian PDF (oval level curves)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("target_2d.png", dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — GP Regression on 1-D Sigmoid                    (TODOs 1–4)
# ═══════════════════════════════════════════════════════════════════════════


def part1_fit_gp_1d():
    X_domain = np.linspace(-2, 2, 200).reshape(-1, 1)
    n_train = 8
    X_train = rng.uniform(-2, 2, size=(n_train, 1))
    noise_std = 0.05
    y_train = sigmoid_1d(X_train).ravel() + rng.randn(n_train) * noise_std

    # ── TODO 1: kernel ──────────────────────────────────────────────────
    kernel = RBF(length_scale=1.0)

    # ── TODO 2: create & fit GP ─────────────────────────────────────────
    # alpha should match the observation noise variance: noise_std² = 0.05² ≈ 0.0025
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.003,
        n_restarts_optimizer=5,
        random_state=RNG_SEED,
    )
    gp.fit(X_train, y_train)

    # ── TODO 3: predict ─────────────────────────────────────────────────
    mu, sigma = gp.predict(X_domain, return_std=True)

    # ── TODO 4: plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(X_domain, sigmoid_1d(X_domain), "k-", lw=1, label="true sigmoid")
    ax.fill_between(
        X_domain.ravel(),
        mu - 2 * sigma,
        mu + 2 * sigma,
        alpha=0.25,
        color="steelblue",
        label="±2σ",
    )
    ax.plot(X_domain, mu, "steelblue", lw=2, label="GP mean")
    ax.scatter(X_train, y_train, c="red", zorder=5, s=40, edgecolors="k", label="train")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Part 1 – GP fit to 1-D sigmoid")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("part1_gp_1d.png", dpi=150)
    plt.show()

    print("Optimised kernel:", gp.kernel_)
    print("Log-marginal-likelihood:", gp.log_marginal_likelihood_value_)


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Active Learning 1-D                             (TODOs 5–6)
# ═══════════════════════════════════════════════════════════════════════════


def ucb_acquisition(gp, X_candidates, kappa=2.0, mu=None, sigma=None):
    # ── TODO 5 ──────────────────────────────────────────────────────────
    if mu is None or sigma is None:
        mu, sigma = gp.predict(X_candidates, return_std=True)
    ucb = mu + kappa * sigma
    best_idx = int(np.argmax(ucb))
    return X_candidates[best_idx].reshape(1, -1), ucb


def active_learning_1d(n_initial=3, n_queries=15, kappa=2.0, noise_std=0.05):
    X_domain = np.linspace(-2, 2, 300).reshape(-1, 1)

    X_train = rng.uniform(-2, 2, size=(n_initial, 1))
    y_train = sigmoid_1d(X_train).ravel() + rng.randn(n_initial) * noise_std

    history = []

    for step in range(n_queries):
        # ── TODO 6 ──────────────────────────────────────────────────────
        kernel = RBF(length_scale=1.0)
        # alpha should match the observation noise variance: noise_std² = 0.05² ≈ 0.0025
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=0.003, n_restarts_optimizer=5, random_state=RNG_SEED
        )
        gp.fit(X_train, y_train)

        mu, sigma = gp.predict(X_domain, return_std=True)
        x_next, acq_values = ucb_acquisition(
            gp, X_domain, kappa=kappa, mu=mu, sigma=sigma
        )
        y_next = sigmoid_1d(x_next).ravel() + rng.randn(1) * noise_std

        history.append(
            {
                "X_train": X_train.copy(),
                "y_train": y_train.copy(),
                "mu": mu.copy(),
                "sigma": sigma.copy(),
                "x_next": x_next.copy(),
            }
        )

        X_train = np.vstack([X_train, x_next])
        y_train = np.append(y_train, y_next)

    return history, X_domain


def plot_active_learning_1d(history, X_domain):
    true_y = sigmoid_1d(X_domain).ravel()
    steps_to_show = [0, 4, 9, len(history) - 1]

    fig, axes = plt.subplots(
        1, len(steps_to_show), figsize=(5 * len(steps_to_show), 4), sharey=True
    )
    for ax, idx in zip(axes, steps_to_show):
        h = history[idx]
        mu, sigma = h["mu"], h["sigma"]
        ax.plot(X_domain, true_y, "k-", lw=1, label="true")
        ax.fill_between(
            X_domain.ravel(),
            mu - 2 * sigma,
            mu + 2 * sigma,
            alpha=0.25,
            color="steelblue",
            label="±2σ",
        )
        ax.plot(X_domain, mu, "steelblue", lw=2, label="GP mean")
        ax.scatter(
            h["X_train"][:, 0], h["y_train"], c="red", zorder=5, s=30, label="obs"
        )
        if "x_next" in h:
            ax.axvline(h["x_next"][0, 0], color="orange", ls="--", lw=1.5, label="next")
        ax.set_title(f"Step {idx}")
        ax.set_xlabel("x")
        if idx == 0:
            ax.set_ylabel("f(x)")
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Part 2 – Active Learning on 1-D Sigmoid (UCB)", fontsize=13)
    plt.tight_layout()
    plt.savefig("part2_active_1d.png", dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# PART 2b — Function Estimation over an Interval (1-D)    (TODOs 7–9)
# ═══════════════════════════════════════════════════════════════════════════


def max_sigma_acquisition(gp, X_candidates):
    # ── TODO 7 ──────────────────────────────────────────────────────────
    mu, sigma = gp.predict(X_candidates, return_std=True)
    best_idx = int(np.argmax(sigma))
    return X_candidates[best_idx].reshape(1, -1), mu, sigma


def function_estimation_1d(
    x_lo=-1.5,
    x_hi=1.5,
    n_initial=3,
    sigma_thresh=0.01,
    max_queries=500,
    noise_std=0.05,
    n_candidates=200,
):
    X_interval = np.linspace(x_lo, x_hi, n_candidates).reshape(-1, 1)

    X_train = rng.uniform(x_lo, x_hi, size=(n_initial, 1))
    y_train = sigmoid_1d(X_train).ravel() + rng.randn(n_initial) * noise_std

    history = []
    converged = False

    for step in range(max_queries):
        # ── TODO 8 ──────────────────────────────────────────────────────────
        kernel = RBF(length_scale=1.0)
        # alpha should match the observation noise variance: noise_std² = 0.05² ≈ 0.0025
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=0.003, n_restarts_optimizer=5, random_state=RNG_SEED
        )
        gp.fit(X_train, y_train)

        x_next, mu, sigma = max_sigma_acquisition(gp, X_interval)
        max_sigma = float(sigma.max())

        if (step % 100) == 0:
            print(f"Step {step}: max σ(x) over interval = {max_sigma:.4f}")

        snap = {
            "X_train": X_train.copy(),
            "y_train": y_train.copy(),
            "mu": mu.copy(),
            "sigma": sigma.copy(),
            "max_sigma_interval": max_sigma,
            "x_next": x_next.copy(),
        }
        history.append(snap)

        if max_sigma <= sigma_thresh:
            converged = True
            break

        y_next = sigmoid_1d(x_next).ravel() + rng.randn(1) * noise_std

        X_train = np.vstack([X_train, x_next])
        y_train = np.append(y_train, y_next)

    return history, X_interval, converged


def plot_function_estimation_1d(history, X_interval, converged, sigma_thresh=0.01):
    # ── TODO 9 ──────────────────────────────────────────────────────────
    x_lo = float(X_interval.min())
    x_hi = float(X_interval.max())
    X_full = np.linspace(-2, 2, 300).reshape(-1, 1)
    y_full_true = sigmoid_1d(X_full).ravel()
    max_sigmas = [h["max_sigma_interval"] for h in history]
    snap_indices = [0, len(history) // 2, len(history) - 1]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Panel 0: convergence curve
    ax = axes[0]
    ax.plot(
        range(len(history)), max_sigmas, "steelblue", lw=2, marker="o", markersize=3
    )
    ax.axhline(
        sigma_thresh, color="red", ls="--", lw=1.5, label=f"threshold {sigma_thresh}"
    )
    if converged:
        ax.axvline(len(history) - 1, color="green", ls=":", lw=1.5, label="converged")
    ax.set_xlabel("Query index")
    ax.set_ylabel("max \u03c3(x) over interval")
    ax.set_title("Max \u03c3(x) over interval vs. query")
    ax.legend(fontsize=8)

    # Panels 1–3: snapshots
    for panel_i, idx in enumerate(snap_indices):
        h = history[idx]
        ax = axes[panel_i + 1]
        mu, sigma = h["mu"], h["sigma"]
        ax.axvspan(x_lo, x_hi, alpha=0.10, color="gold", label="target interval")
        ax.plot(X_full.ravel(), y_full_true, "k-", lw=1, label="true")
        ax.fill_between(
            X_interval.ravel(),
            mu - 2 * sigma,
            mu + 2 * sigma,
            alpha=0.25,
            color="steelblue",
        )
        ax.plot(X_interval.ravel(), mu, "steelblue", lw=2, label="GP mean")
        ax.scatter(
            h["X_train"][:, 0], h["y_train"], c="red", zorder=5, s=25, label="obs"
        )
        if "x_next" in h:
            ax.axvline(h["x_next"][0, 0], color="orange", ls="--", lw=1.5, label="next")
        ax.set_title(f"Step {idx}  (max \u03c3 = {h['max_sigma_interval']:.3f})")
        ax.set_xlabel("x")
        if panel_i == 0:
            ax.set_ylabel("f(x)")
            ax.legend(fontsize=7, loc="upper left")

    status = "converged" if converged else "did not converge"
    fig.suptitle(
        f"Part 2b \u2013 Function Estimation over [{x_lo}, {x_hi}]  "
        f"({status}, {len(history)} queries)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("part2b_estimation_1d.png", dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — 2-D Active Learning                            (TODOs 10–11)
# ═══════════════════════════════════════════════════════════════════════════


def make_2d_grid(xlim=(-3, 3), ylim=(-3, 3), n=50):
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    Xg, Yg = np.meshgrid(xs, ys)
    grid_flat = np.column_stack([Xg.ravel(), Yg.ravel()])
    return Xg, Yg, grid_flat


def active_learning_2d(
    n_initial=5,
    n_queries=30,
    kappa=2.0,
    noise_std=0.01,
    sigma_x=1.0,
    sigma_y=0.5,
    grid_n=50,
):
    Xg, Yg, grid_flat = make_2d_grid(n=grid_n)

    # ── TODO 10: initial observations ────────────────────────────────────
    X_train = rng.uniform(-3, 3, size=(n_initial, 2))
    y_train = (
        inverted_gaussian_2d(X_train, sigma_x, sigma_y)
        + rng.randn(n_initial) * noise_std
    )

    history = []

    for step in range(n_queries):
        # ── TODO 11: 2-D loop body ──────────────────────────────────────
        # Fix kernel hyperparameters to domain knowledge: ARD Matérn with
        # length scales matching the target's σ_x / σ_y so uncertainty maps
        # are always oval-shaped.  Using optimizer=None avoids the
        # marginal-likelihood optimisation that, with few points in [-3,3]²,
        # drives one or both length scales to degenerate values → stripes.
        kernel = Matern(
            length_scale=[sigma_x, sigma_y],
            length_scale_bounds="fixed",
            nu=2.5,
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise_std**2,
            optimizer=None,
            random_state=RNG_SEED,
        )
        gp.fit(X_train, y_train)

        mu, sigma = gp.predict(grid_flat, return_std=True)
        x_next, _ = ucb_acquisition(gp, grid_flat, kappa=kappa, mu=mu, sigma=sigma)
        y_next = (
            inverted_gaussian_2d(x_next, sigma_x, sigma_y).ravel()
            + rng.randn(1) * noise_std
        )

        history.append(
            {
                "X_train": X_train.copy(),
                "y_train": y_train.copy(),
                "mu_grid": mu.reshape(Xg.shape),
                "sigma_grid": sigma.reshape(Xg.shape),
                "x_next": x_next.copy(),
            }
        )

        X_train = np.vstack([X_train, x_next])
        y_train = np.append(y_train, y_next)

        if (step + 1) % 10 == 0:
            print(f"  2-D active learning step {step + 1}/{n_queries}")

    return history, Xg, Yg


def plot_active_learning_2d_snapshots(history, Xg, Yg, sigma_x=1.0, sigma_y=0.5):
    Z_true = inverted_gaussian_2d(
        np.column_stack([Xg.ravel(), Yg.ravel()]), sigma_x, sigma_y
    ).reshape(Xg.shape)

    steps = [0, len(history) // 3, 2 * len(history) // 3, len(history) - 1]
    fig, axes = plt.subplots(2, len(steps), figsize=(5 * len(steps), 8))

    for col, idx in enumerate(steps):
        h = history[idx]
        ax = axes[0, col]
        ax.contourf(Xg, Yg, h["mu_grid"], levels=30, cmap="viridis")
        ax.contour(
            Xg, Yg, Z_true, levels=8, colors="w", linewidths=0.5, linestyles="--"
        )
        ax.scatter(
            h["X_train"][:, 0],
            h["X_train"][:, 1],
            c="red",
            s=15,
            edgecolors="k",
            lw=0.4,
        )
        ax.set_title(f"GP mean – step {idx}")
        ax.set_aspect("equal")

        ax = axes[1, col]
        ax.contourf(Xg, Yg, h["sigma_grid"], levels=30, cmap="magma")
        ax.scatter(
            h["X_train"][:, 0],
            h["X_train"][:, 1],
            c="cyan",
            s=15,
            edgecolors="k",
            lw=0.4,
        )
        if "x_next" in h:
            ax.scatter(
                h["x_next"][0, 0],
                h["x_next"][0, 1],
                marker="*",
                s=200,
                c="lime",
                edgecolors="k",
            )
        ax.set_title(f"σ(x) – step {idx}")
        ax.set_aspect("equal")

    fig.suptitle("Part 3 – Active Learning on 2-D Inverted Gaussian", fontsize=14)
    plt.tight_layout()
    plt.savefig("part3_active_2d.png", dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# PART 3b — When does kernel learning fail?              (TODOs 12–13)
#
# Lesson: marginal-likelihood optimisation of ARD length scales (empirical
# Bayes / Type II ML, Murphy §15.3.2 / Bishop §6.4.3) requires enough data
# to constrain the per-axis posterior.  With only ~5 points in [-3,3]² the
# likelihood is nearly flat along an under-sampled axis, so the optimiser
# pushes that length scale to its upper bound — producing horizontal or
# vertical *stripes* in the uncertainty map instead of the true oval shape.
# Increasing n_initial to ~25 gives the optimiser enough cross-axis coverage
# to identify both length scales correctly, and the ovals emerge naturally
# without any hand-tuning.
#
# This motivates the design choice in Part 3 (fixing length scales to domain
# knowledge).  Using more initial data is the other valid fix.
# ═══════════════════════════════════════════════════════════════════════════


def kernel_sensitivity_2d(
    n_initial=5,
    n_queries=15,
    kappa=2.0,
    noise_std=0.01,
    sigma_x=1.0,
    sigma_y=0.5,
    grid_n=50,
    X_initial=None,
):
    """Run active learning with a freely-optimised ARD Matérn kernel.

    Unlike Part 3, the length scales are NOT fixed — they are learned by
    maximising the log marginal likelihood.  Call this function twice with
    different n_initial values to illustrate the failure mode.

    Pass X_initial to control the starting design.  For the sparse failure
    mode, supply points confined to a line (e.g. all at y≈0): the GP then
    has no information about between-row variation and the optimiser pushes
    ℓ_y to its upper bound, producing clear horizontal stripes.
    """
    Xg, Yg, grid_flat = make_2d_grid(n=grid_n)

    if X_initial is not None:
        X_train = np.asarray(X_initial, dtype=float)
        n_initial = len(X_train)
    else:
        X_train = rng.uniform(-3, 3, size=(n_initial, 2))
    y_train = (
        inverted_gaussian_2d(X_train, sigma_x, sigma_y)
        + rng.randn(n_initial) * noise_std
    )

    history = []

    for step in range(n_queries):
        # ── TODO 12 ─────────────────────────────────────────────────────────
        # Same structure as TODO 11, but use a *freely-optimised* ARD Matérn.
        # Do NOT pass optimizer=None — the whole point is to let the marginal
        # likelihood decide the length scales.
        #
        # After fitting, extract per-axis length scales with:
        #   ls = gp.kernel_.k2.length_scale
        # and store them in the snapshot so the plot can annotate them.
        kernel = C(1.0, (1e-2, 1e2)) * Matern(
            length_scale=[1.0, 1.0],
            length_scale_bounds=(0.1, 20.0),
            nu=2.5,
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise_std**2,
            n_restarts_optimizer=3,
            random_state=RNG_SEED,
        )
        gp.fit(X_train, y_train)

        mu, sigma = gp.predict(grid_flat, return_std=True)
        x_next, _ = ucb_acquisition(gp, grid_flat, kappa=kappa, mu=mu, sigma=sigma)
        y_next = (
            inverted_gaussian_2d(x_next, sigma_x, sigma_y).ravel()
            + rng.randn(1) * noise_std
        )

        ls = gp.kernel_.k2.length_scale  # ARD length scales after optimisation
        history.append(
            {
                "X_train": X_train.copy(),
                "y_train": y_train.copy(),
                "mu_grid": mu.reshape(Xg.shape),
                "sigma_grid": sigma.reshape(Xg.shape),
                "x_next": x_next.copy(),
                "length_scales": ls.copy(),
            }
        )

        X_train = np.vstack([X_train, x_next])
        y_train = np.append(y_train, y_next)

    return history, Xg, Yg


def plot_kernel_sensitivity_2d(
    history_sparse,
    history_dense,
    Xg,
    Yg,
    n_initial_sparse=5,
    n_initial_dense=25,
):
    """Compare uncertainty maps with sparse vs dense initial data.

    Top row:    n_initial_sparse  (likely to show stripe artefacts).
    Bottom row: n_initial_dense   (should recover oval uncertainty maps).
    Each column is a different step; titles show the optimised length scales.
    """
    # ── TODO 13 ─────────────────────────────────────────────────────────────
    snap_indices = [0, 5, 10, len(history_sparse) - 1]
    fig, axes = plt.subplots(2, len(snap_indices), figsize=(5 * len(snap_indices), 8))

    for col, idx in enumerate(snap_indices):
        for row, (history, n_init) in enumerate(
            [(history_sparse, n_initial_sparse), (history_dense, n_initial_dense)]
        ):
            hidx = min(idx, len(history) - 1)
            h = history[hidx]
            ax = axes[row, col]
            ax.contourf(Xg, Yg, h["sigma_grid"], levels=20, cmap="magma")
            ax.scatter(
                h["X_train"][:, 0],
                h["X_train"][:, 1],
                c="cyan",
                s=15,
                edgecolors="k",
                lw=0.4,
            )
            ls = h["length_scales"]
            ax.set_title(
                f"step {hidx}\nℓ_x={ls[0]:.2f},  ℓ_y={ls[1]:.2f}",
                fontsize=8,
            )
            ax.set_aspect("equal")

    axes[0, 0].set_ylabel(f"Sparse  n_initial={n_initial_sparse}\nσ(x)", fontsize=9)
    axes[1, 0].set_ylabel(f"Dense   n_initial={n_initial_dense}\nσ(x)", fontsize=9)
    fig.suptitle(
        "Part 3b – ARD kernel learning: sparse vs dense initial data\n"
        "Stripe artefacts (top row, early steps) arise because the\n"
        "marginal-likelihood optimiser cannot identify both length scales\n"
        "from only a few points.  More initial data (bottom row) fixes it.",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("part3b_kernel_sensitivity.png", dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — Animation                                      (TODOs 14–16)
# ═══════════════════════════════════════════════════════════════════════════


def animate_2d(history, Xg, Yg, sigma_x=1.0, sigma_y=0.5, save_gif=True):
    Z_true = inverted_gaussian_2d(
        np.column_stack([Xg.ravel(), Yg.ravel()]), sigma_x, sigma_y
    ).reshape(Xg.shape)

    fig, (ax_mean, ax_unc) = plt.subplots(1, 2, figsize=(12, 5))

    all_mu = np.concatenate([h["mu_grid"].ravel() for h in history])
    all_sigma = np.concatenate([h["sigma_grid"].ravel() for h in history])
    mu_lim = (all_mu.min(), all_mu.max())
    sigma_lim = (all_sigma.min(), all_sigma.max())

    # ── TODO 14: init ────────────────────────────────────────────────────
    def init():
        ax_mean.clear()
        ax_unc.clear()
        return []

    # ── TODO 15: update ─────────────────────────────────────────────────
    def update(frame):
        h = history[frame]

        # Left panel – GP mean
        ax_mean.clear()
        ax_mean.contourf(
            Xg,
            Yg,
            h["mu_grid"],
            levels=30,
            cmap="viridis",
            vmin=mu_lim[0],
            vmax=mu_lim[1],
        )
        ax_mean.contour(
            Xg, Yg, Z_true, levels=8, colors="w", linewidths=0.5, linestyles="--"
        )
        ax_mean.scatter(
            h["X_train"][:, 0],
            h["X_train"][:, 1],
            c="red",
            s=18,
            edgecolors="k",
            lw=0.4,
        )
        ax_mean.set_title(f"GP mean – step {frame}")
        ax_mean.set_aspect("equal")

        # Right panel – uncertainty
        ax_unc.clear()
        ax_unc.contourf(
            Xg,
            Yg,
            h["sigma_grid"],
            levels=30,
            cmap="magma",
            vmin=sigma_lim[0],
            vmax=sigma_lim[1],
        )
        ax_unc.scatter(
            h["X_train"][:, 0],
            h["X_train"][:, 1],
            c="cyan",
            s=18,
            edgecolors="k",
            lw=0.4,
        )
        if "x_next" in h:
            ax_unc.scatter(
                h["x_next"][0, 0],
                h["x_next"][0, 1],
                marker="*",
                s=250,
                c="lime",
                edgecolors="k",
                lw=1,
            )
        ax_unc.set_title(f"Uncertainty σ(x) – step {frame}")
        ax_unc.set_aspect("equal")

        fig.suptitle("Active Learning – 2-D Inverted Gaussian", fontsize=13)
        return []

    # ── TODO 16: create animation ───────────────────────────────────────────────
    anim = FuncAnimation(
        fig,
        update,
        frames=len(history),
        init_func=init,
        interval=500,
        repeat=True,
    )

    if save_gif:
        anim.save("active_learning_2d.gif", writer="pillow", fps=2)
        print("Saved active_learning_2d.gif")

    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Part 0 – Plot the 2-D target function")
    print("=" * 60)
    plot_2d_target()

    print("\n" + "=" * 60)
    print("Part 1 – GP fit to 1-D sigmoid")
    print("=" * 60)
    part1_fit_gp_1d()

    print("\n" + "=" * 60)
    print("Part 2 – Active learning on 1-D sigmoid")
    print("=" * 60)
    history_1d, X_domain_1d = active_learning_1d()
    plot_active_learning_1d(history_1d, X_domain_1d)

    print("\n" + "=" * 60)
    print("Part 2b – Function estimation over an interval (1-D)")
    print("=" * 60)
    history_est, X_interval, converged = function_estimation_1d()
    plot_function_estimation_1d(history_est, X_interval, converged)

    print("\n" + "=" * 60)
    print("Part 3 – Active learning on 2-D inverted Gaussian")
    print("=" * 60)
    history_2d, Xg, Yg = active_learning_2d()
    plot_active_learning_2d_snapshots(history_2d, Xg, Yg)

    print("\n" + "=" * 60)
    print("Part 3b – Kernel learning failure modes (2-D)")
    print("=" * 60)
    print("  Running with n_initial=5, collinear (sparse — expect stripes)...")
    # Initial points all near y=0: the GP has zero information about y-variation
    # so the optimiser pushes ℓ_y to its upper bound → horizontal stripes.
    X_init_sparse = np.column_stack(
        [
            rng.uniform(-3, 3, 5),
            rng.uniform(-0.3, 0.3, 5),
        ]
    )
    history_sparse, Xg_s, Yg_s = kernel_sensitivity_2d(
        n_initial=5, X_initial=X_init_sparse
    )
    print("  Running with n_initial=25 (dense — expect ovals)...")
    history_dense, _, _ = kernel_sensitivity_2d(n_initial=25)
    plot_kernel_sensitivity_2d(history_sparse, history_dense, Xg_s, Yg_s)

    print("\n" + "=" * 60)
    print("Part 4 – Animation")
    print("=" * 60)
    animate_2d(history_2d, Xg, Yg, save_gif=True)

    print("\nDone!  Check the saved .png and .gif files.")
