#!/usr/bin/env python3
"""
============================================================================
Homework: Active Learning with Gaussian Processes
============================================================================

OVERVIEW
--------
In this assignment you will build up an active learning system powered by
Gaussian Process (GP) regression.  The roadmap:

  Part 1 – Fit a GP to noisy samples of a 1-D sigmoid and plot the
           posterior mean ± uncertainty.
  Part 2 – Implement Upper-Confidence-Bound (UCB) acquisition and run an
           active-learning loop on the 1-D sigmoid.
  Part 3 – Move to 2-D: learn an inverted Gaussian PDF surface with oval
           (anisotropic) level curves.
  Part 4 – Animate the 2-D exploration so you can watch the GP posterior
           evolve as new points are acquired.

BACKGROUND REFRESHER  (skim if rusty – Murphy Ch. 15 / Bishop §6.4)
-------------------
A Gaussian Process defines a distribution over functions:
    f(x) ~ GP(m(x), k(x, x'))
where m(x) is the mean function (often 0) and k is the covariance (kernel)
function.  Given observations (X, y) with noise variance σ²_n, the posterior
predictive at a test point x* is Gaussian:

    μ(x*) = k(x*, X) [K(X,X) + σ²_n I]⁻¹ y
    σ²(x*) = k(x*, x*) - k(x*, X) [K(X,X) + σ²_n I]⁻¹ k(X, x*)

The kernel encodes assumptions about smoothness, lengthscale, etc.
sklearn.gaussian_process wraps all of this for us.

ACTIVE LEARNING refresher
-------------------------
Instead of choosing training points at random we *choose* the next point
to query the true function.  A common heuristic is
Upper Confidence Bound (UCB):

    a(x) = μ(x) + κ · σ(x)

where κ (kappa) controls exploration vs. exploitation.  We pick the x that
maximises a(x) among a set of candidates.

INSTRUCTIONS
------------
1.  Search for TODO markers – each one asks you to fill in code.
2.  There are 11 TODOs total, roughly grouped by part.
3.  Run the script and confirm the figures / animations look sensible.
4.  Expected time: ~2 hours for someone comfortable with numpy, matplotlib,
    and the theory of GPs.

DEPENDENCIES
------------
    pip install numpy matplotlib scikit-learn

    (optional, for Part 4 animation export)
    pip install pillow          # for saving .gif
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
# TARGET FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def sigmoid_1d(x, slope=5.0, shift=0.0):
    """Sigmoid: f(x) = 1 / (1 + exp(-slope * (x - shift)))."""
    return 1.0 / (1.0 + np.exp(-slope * (x - shift)))


def inverted_gaussian_2d(X, sigma_x=1.0, sigma_y=0.5):
    """
    Inverted (negated) 2-D Gaussian PDF centred at the origin.

    f(x, y) = - (1 / (2π σ_x σ_y)) exp(-0.5 (x²/σ_x² + y²/σ_y²))

    Because σ_x ≠ σ_y the level curves are *ovals* (axis-aligned ellipses).

    Parameters
    ----------
    X : array-like, shape (n, 2)
        Each row is an (x, y) point.
    sigma_x, sigma_y : float
        Standard deviations along x and y.

    Returns
    -------
    f : ndarray, shape (n,)
    """
    X = np.asarray(X)
    x, y = X[:, 0], X[:, 1]
    norm = 1.0 / (2.0 * np.pi * sigma_x * sigma_y)
    exponent = -0.5 * (x**2 / sigma_x**2 + y**2 / sigma_y**2)
    return -norm * np.exp(exponent)


# Quick sanity-check plot of the 2-D target (run this cell to make sure the
# function looks right before you start implementing anything).
def plot_2d_target(sigma_x=1.0, sigma_y=0.5, grid_n=80):
    """Contour plot of the inverted Gaussian target."""
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
# PART 1 — Gaussian Process Regression on a 1-D Sigmoid
# ═══════════════════════════════════════════════════════════════════════════
#
# Goal: Fit a GP to a handful of noisy sigmoid observations and visualise
#       the posterior mean and ±2σ uncertainty band.
#
# Recall from Murphy §15.2 / Bishop §6.4:
#   • The kernel controls our prior beliefs about the function.
#   • The RBF (squared-exponential) kernel
#         k(x, x') = σ_f² exp( -||x - x'||² / (2 ℓ²) )
#     assumes smooth functions.  σ_f is the signal variance, ℓ the
#     lengthscale.
#   • Adding a WhiteKernel accounts for observation noise σ²_n.
#   • sklearn optimises kernel hyperparameters by maximising the log
#     marginal likelihood (Murphy Eq. 15.2.31).
# ═══════════════════════════════════════════════════════════════════════════


def part1_fit_gp_1d():
    """Fit a GP to noisy 1-D sigmoid samples and plot the posterior."""

    # --- data ----------------------------------------------------------
    X_domain = np.linspace(-2, 2, 200).reshape(-1, 1)  # dense test grid
    n_train = 8
    X_train = rng.uniform(-2, 2, size=(n_train, 1))
    noise_std = 0.05
    y_train = sigmoid_1d(X_train).ravel() + rng.randn(n_train) * noise_std

    # --- TODO 1 --------------------------------------------------------
    # Define an sklearn kernel suitable for learning a smooth 1-D function
    # from noisy observations.  A good starting point:
    #
    #   kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
    #
    # Hint: C(1.0) is a ConstantKernel that models the signal variance σ_f².
    #       RBF(...) is the squared-exponential.
    #       WhiteKernel models i.i.d. observation noise.
    #
    # WRITE YOUR CODE BELOW (1-2 lines)
    # kernel = ???
    raise NotImplementedError("TODO 1: define `kernel`")

    # --- TODO 2 --------------------------------------------------------
    # Create a GaussianProcessRegressor with the kernel above.
    #   • Set n_restarts_optimizer to something > 0 so the marginal-
    #     likelihood optimisation doesn't get stuck in a local optimum.
    #   • Set random_state=RNG_SEED for reproducibility.
    #
    # Then fit (.fit) it to X_train, y_train.
    #
    # WRITE YOUR CODE BELOW (2-3 lines)
    # gp = GaussianProcessRegressor(...)
    # gp.fit(...)
    raise NotImplementedError("TODO 2: create and fit the GP")

    # --- TODO 3 --------------------------------------------------------
    # Use gp.predict to get the posterior mean and standard deviation on
    # X_domain.  (Hint: set return_std=True.)
    #
    # WRITE YOUR CODE BELOW (1 line)
    # mu, sigma = ???
    raise NotImplementedError("TODO 3: predict on X_domain")

    # --- TODO 4 --------------------------------------------------------
    # Plot:
    #   1. The true sigmoid curve (thin black line).
    #   2. A shaded band for μ ± 2σ  (use ax.fill_between).
    #   3. The GP posterior mean (coloured line).
    #   4. The training points (scatter).
    # Label axes (x, f(x)), add a legend, and add the title
    # "Part 1 – GP fit to 1-D sigmoid".
    #
    # WRITE YOUR CODE BELOW (~10-15 lines)
    raise NotImplementedError("TODO 4: plot the GP posterior")

    # Save your figure
    plt.savefig("part1_gp_1d.png", dpi=150)
    plt.show()

    # Print the optimised kernel so you can inspect the learned
    # hyperparameters.  Compare them to the true noise_std and the
    # sigmoid's effective lengthscale.
    print("Optimised kernel:", gp.kernel_)
    print("Log-marginal-likelihood:", gp.log_marginal_likelihood_value_)


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Active Learning Loop (1-D)
# ═══════════════════════════════════════════════════════════════════════════
#
# Goal: Starting from a small random training set, repeatedly choose the
#       next query point using UCB, observe the function there, add it to
#       the training set, and refit the GP.
#
# After the loop, compare the fit obtained by active sampling to one with
# the same number of random samples.
#
# Recall:
#   UCB:  a(x) = μ(x) + κ σ(x),    κ > 0
#
#   The acquisition function balances *exploitation* (high μ) with
#   *exploration* (high σ).  For pure exploration of the whole function
#   (as opposed to optimisation), you might set κ large, or even just
#   use σ(x) alone (maximum uncertainty sampling).
# ═══════════════════════════════════════════════════════════════════════════


def ucb_acquisition(gp, X_candidates, kappa=2.0):
    """
    Compute UCB acquisition values and return the best candidate.

    Parameters
    ----------
    gp : fitted GaussianProcessRegressor
    X_candidates : ndarray, shape (n_cand, d)
    kappa : float
        Exploration weight.

    Returns
    -------
    x_next : ndarray, shape (1, d)
        The candidate with highest UCB.
    acq_values : ndarray, shape (n_cand,)
        UCB values for all candidates (useful for plotting).
    """
    # --- TODO 5 --------------------------------------------------------
    # 1. Use gp.predict with return_std=True on X_candidates.
    # 2. Compute UCB = mu + kappa * sigma.
    # 3. Find the index of the maximum UCB value.
    # 4. Return X_candidates[best_idx] reshaped to (1, d), and the full
    #    UCB array.
    #
    # WRITE YOUR CODE BELOW (~4 lines)
    raise NotImplementedError("TODO 5: implement UCB acquisition")


def active_learning_1d(n_initial=3, n_queries=15, kappa=2.0, noise_std=0.05):
    """
    Active learning loop on the 1-D sigmoid.

    Returns
    -------
    history : list of dict
        Each entry has keys: 'X_train', 'y_train', 'mu', 'sigma'
        (one per iteration, so you can animate later).
    """
    X_domain = np.linspace(-2, 2, 300).reshape(-1, 1)

    # Initial random observations
    X_train = rng.uniform(-2, 2, size=(n_initial, 1))
    y_train = sigmoid_1d(X_train).ravel() + rng.randn(n_initial) * noise_std

    history = []

    for step in range(n_queries):

        # --- TODO 6 ----------------------------------------------------
        # 1. Define a kernel (same style as Part 1 is fine).
        # 2. Create and fit a GaussianProcessRegressor to X_train, y_train.
        # 3. Predict on X_domain to get mu, sigma.
        # 4. Call ucb_acquisition(gp, X_domain, kappa) to pick x_next.
        # 5. Observe y_next = sigmoid_1d(x_next) + noise.
        # 6. Append x_next, y_next to X_train, y_train.
        #
        # Also append a snapshot to `history`:
        #   history.append({
        #       'X_train': X_train.copy(),
        #       'y_train': y_train.copy(),
        #       'mu': mu.copy(),
        #       'sigma': sigma.copy(),
        #       'x_next': x_next.copy(),
        #   })
        #
        # WRITE YOUR CODE BELOW (~12-18 lines)
        raise NotImplementedError("TODO 6: active learning 1-D loop body")

    return history, X_domain


def plot_active_learning_1d(history, X_domain):
    """Plot selected snapshots of the 1-D active learning run."""
    true_y = sigmoid_1d(X_domain).ravel()
    steps_to_show = [0, 4, 9, len(history) - 1]  # first, middle, last

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
            h["X_train"][:, 0],
            h["y_train"],
            c="red",
            zorder=5,
            s=30,
            label="observations",
        )
        if "x_next" in h:
            ax.axvline(
                h["x_next"][0, 0], color="orange", ls="--", lw=1.5, label="next query"
            )
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
# PART 3 — 2-D Active Learning on the Inverted Gaussian
# ═══════════════════════════════════════════════════════════════════════════
#
# The target is inverted_gaussian_2d with σ_x = 1.0, σ_y = 0.5.
# The domain is [-3, 3] × [-3, 3].
#
# Key change from 1-D:
#   • Candidate points live on a 2-D grid (or quasi-random set).
#   • Visualisation uses contour plots instead of line plots.
#   • The Matérn kernel can be a good choice in higher-D (see
#     Murphy §15.2.1, or sklearn docs).
# ═══════════════════════════════════════════════════════════════════════════


def make_2d_grid(xlim=(-3, 3), ylim=(-3, 3), n=50):
    """Return a meshgrid for evaluation and a flat (n*n, 2) array."""
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
    """
    Active learning loop on the 2-D inverted Gaussian.

    Returns
    -------
    history : list of dict
    Xg, Yg  : meshgrid arrays (for plotting)
    """
    Xg, Yg, grid_flat = make_2d_grid(n=grid_n)

    # --- TODO 7 --------------------------------------------------------
    # Generate n_initial random 2-D points in [-3, 3]² and observe the
    # target function (with noise).
    #
    # WRITE YOUR CODE BELOW (3-4 lines)
    # X_train = ???
    # y_train = ???
    raise NotImplementedError("TODO 7: initial 2-D observations")

    history = []

    for step in range(n_queries):

        # --- TODO 8 ----------------------------------------------------
        # Same structure as the 1-D loop (TODO 6), but now in 2-D:
        #   1. Choose a kernel.  Try Matern(nu=2.5) or RBF — optionally
        #      with separate length_scale per dimension:
        #        RBF(length_scale=[1.0, 1.0])
        #      sklearn will optimise them independently (ARD).
        #   2. Fit the GP.
        #   3. Predict on grid_flat → mu, sigma.
        #   4. UCB acquisition.
        #   5. Observe the new point and append.
        #   6. Save a history snapshot (include mu, sigma reshaped to grid).
        #
        # history.append({
        #     'X_train': X_train.copy(),
        #     'y_train': y_train.copy(),
        #     'mu_grid': mu.reshape(Xg.shape),
        #     'sigma_grid': sigma.reshape(Xg.shape),
        #     'x_next': x_next.copy(),
        # })
        #
        # WRITE YOUR CODE BELOW (~15-20 lines)
        raise NotImplementedError("TODO 8: active learning 2-D loop body")

    return history, Xg, Yg


def plot_active_learning_2d_snapshots(history, Xg, Yg, sigma_x=1.0, sigma_y=0.5):
    """Plot a 2×4 grid: top row GP mean, bottom row uncertainty."""
    Z_true = inverted_gaussian_2d(
        np.column_stack([Xg.ravel(), Yg.ravel()]), sigma_x, sigma_y
    ).reshape(Xg.shape)

    steps = [0, len(history) // 3, 2 * len(history) // 3, len(history) - 1]
    fig, axes = plt.subplots(2, len(steps), figsize=(5 * len(steps), 8))

    for col, idx in enumerate(steps):
        h = history[idx]
        # Top: GP mean vs. true
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

        # Bottom: uncertainty
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
# PART 4 — Animation
# ═══════════════════════════════════════════════════════════════════════════
#
# Create an animated figure that cycles through the history snapshots.
# Each frame shows:
#   • Left panel:  GP posterior mean contour with training points.
#   • Right panel: GP uncertainty contour with the next-query star.
#
# matplotlib.animation.FuncAnimation is the tool of choice here.
# ═══════════════════════════════════════════════════════════════════════════


def animate_2d(history, Xg, Yg, sigma_x=1.0, sigma_y=0.5, save_gif=True):
    """
    Animate the 2-D active learning run.

    Parameters
    ----------
    history, Xg, Yg : as returned by active_learning_2d
    save_gif : bool
        If True, save the animation to 'active_learning_2d.gif'.
    """
    Z_true = inverted_gaussian_2d(
        np.column_stack([Xg.ravel(), Yg.ravel()]), sigma_x, sigma_y
    ).reshape(Xg.shape)

    fig, (ax_mean, ax_unc) = plt.subplots(1, 2, figsize=(12, 5))

    # We need to fix colour limits so the animation is smooth.
    all_mu = np.concatenate([h["mu_grid"].ravel() for h in history])
    all_sigma = np.concatenate([h["sigma_grid"].ravel() for h in history])
    mu_lim = (all_mu.min(), all_mu.max())
    sigma_lim = (all_sigma.min(), all_sigma.max())

    # --- TODO 9 --------------------------------------------------------
    # Implement the `init` function for FuncAnimation.
    #   • Clear both axes.
    #   • Return an empty list of artists.
    #
    # WRITE YOUR CODE BELOW (~4 lines)
    def init():
        raise NotImplementedError("TODO 9: animation init")

    # --- TODO 10 -------------------------------------------------------
    # Implement the `update(frame)` function for FuncAnimation.
    #
    #   frame : int  (index into history)
    #
    # For each frame:
    #   1. Clear both axes.
    #   2. Left axis (ax_mean):
    #        contourf of history[frame]['mu_grid']  (use mu_lim for vmin/vmax)
    #        contour of Z_true in white dashed lines
    #        scatter of training points in red
    #        title: "GP mean – step {frame}"
    #   3. Right axis (ax_unc):
    #        contourf of history[frame]['sigma_grid'] (use sigma_lim, cmap='magma')
    #        scatter of training points in cyan
    #        star marker at x_next in lime green (if present)
    #        title: "Uncertainty σ(x) – step {frame}"
    #   4. Set aspect='equal' on both axes.
    #   5. Return [] (blitting not needed with clear-and-redraw).
    #
    # WRITE YOUR CODE BELOW (~20-25 lines)
    def update(frame):
        raise NotImplementedError("TODO 10: animation update function")

    # --- TODO 11 -------------------------------------------------------
    # Create the FuncAnimation and (optionally) save it.
    #
    #   anim = FuncAnimation(fig, update, frames=len(history),
    #                        init_func=init, interval=500, repeat=True)
    #
    #   if save_gif:
    #       anim.save('active_learning_2d.gif', writer='pillow', fps=2)
    #       print("Saved active_learning_2d.gif")
    #
    #   plt.show()
    #
    # WRITE YOUR CODE BELOW (~5 lines)
    raise NotImplementedError("TODO 11: create and display animation")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN – run each part in order
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
    print("Part 3 – Active learning on 2-D inverted Gaussian")
    print("=" * 60)
    history_2d, Xg, Yg = active_learning_2d()
    plot_active_learning_2d_snapshots(history_2d, Xg, Yg)

    print("\n" + "=" * 60)
    print("Part 4 – Animation")
    print("=" * 60)
    animate_2d(history_2d, Xg, Yg, save_gif=True)

    print("\nDone!  Check the saved .png and .gif files.")
