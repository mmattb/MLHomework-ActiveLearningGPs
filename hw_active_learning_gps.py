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
  Part 2b– Function estimation: query until uncertainty over a chosen
           interval [x_lo, x_hi] drops below a threshold σ_thresh.
  Part 3 – Move to 2-D: learn an inverted Gaussian PDF surface with oval
           (anisotropic) level curves.
  Part 3b– Explore when kernel hyperparameter learning fails: run the same
           2-D loop with an ARD Matérn whose length scales are *not* fixed,
           compare sparse (n_initial=5) vs dense (n_initial=25) and observe
           the stripe artefacts that appear with too few initial points.
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
2.  There are 14 TODOs total, roughly grouped by part.
3.  Run the script and confirm the figures / animations look sensible.
4.  Expected time: ~2-3 hours for someone comfortable with numpy, matplotlib,
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


def ucb_acquisition(gp, X_candidates, kappa=2.0, mu=None, sigma=None):
    """
    Compute UCB acquisition values and return the best candidate.

    Parameters
    ----------
    gp : fitted GaussianProcessRegressor
    X_candidates : ndarray, shape (n_cand, d)
    kappa : float
        Exploration weight.
    mu : ndarray, shape (n_cand,), optional
        Pre-computed posterior mean. If None, will be predicted.
    sigma : ndarray, shape (n_cand,), optional
        Pre-computed posterior std. If None, will be predicted.

    Returns
    -------
    x_next : ndarray, shape (1, d)
        The candidate with highest UCB.
    acq_values : ndarray, shape (n_cand,)
        UCB values for all candidates (useful for plotting).
    """
    # --- TODO 5 --------------------------------------------------------
    # 1. If mu/sigma are not provided, predict them on X_candidates.
    # 2. Compute UCB = mu + kappa * sigma.
    # 3. Find the index of the maximum UCB value.
    # 4. Return X_candidates[best_idx] reshaped to (1, d), and the full UCB array.
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
# PART 2b — Function Estimation over an Interval (1-D)
# ═══════════════════════════════════════════════════════════════════════════
#
# Goal: Estimate the sigmoid over a *specific interval* [x_lo, x_hi] to a
#       prescribed uncertainty budget σ_thresh.  Instead of running for a
#       fixed number of steps, keep querying until:
#
#           max_{x ∈ [x_lo, x_hi]} σ(x)  ≤  σ_thresh
#
#       The key conceptual shift compared with Part 2:
#
#         Part 2 objective  →  find a high-value region   (UCB: μ + κσ)
#         Part 2b objective →  characterise the function everywhere over
#                               a chosen interval          (pure max-σ: argmax σ)
#
#       When you only care about a sub-interval you still benefit from
#       observations outside it (they constrain the kernel fit), but for
#       simplicity here we restrict the candidate set to [x_lo, x_hi].
#
# Think about:
#   • Why does pure max-σ acquisition tend to place observations roughly
#     evenly across the interval?
#     (Hint: for an RBF kernel, how does σ(x) look between two nearby
#      observations?  Where is it largest?)
#   • How does the number of queries needed scale with σ_thresh?
#     Try σ_thresh = 0.05, 0.02, 0.01 and record the query counts.
#   • What changes if you widen the interval from [-1.5, 1.5] to [-2, 2]?
# ═══════════════════════════════════════════════════════════════════════════


def max_sigma_acquisition(gp, X_candidates):
    """
    Pure maximum-uncertainty acquisition: pick the candidate with the
    largest posterior standard deviation.

    This is equivalent to UCB with κ → ∞, or to the greedy strategy
    that minimises the worst-case posterior variance at each step.

    Parameters
    ----------
    gp : fitted GaussianProcessRegressor
    X_candidates : ndarray, shape (n_cand, 1)

    Returns
    -------
    x_next : ndarray, shape (1, 1)
        The candidate with highest sigma.
    mu : ndarray, shape (n_cand,)
        Posterior mean on all candidates.
    sigma : ndarray, shape (n_cand,)
        Posterior std on all candidates.
    """
    # --- TODO 7 --------------------------------------------------------
    # 1. Predict both mu and sigma on X_candidates (return_std=True).
    # 2. Find the index of the maximum sigma value.
    # 3. Return (X_candidates[best_idx].reshape(1, 1), mu, sigma).
    #
    # WRITE YOUR CODE BELOW (~3 lines)
    raise NotImplementedError("TODO 7: implement max-sigma acquisition")


def function_estimation_1d(
    x_lo=-1.5,
    x_hi=1.5,
    n_initial=3,
    sigma_thresh=0.01,
    max_queries=500,
    noise_std=0.05,
    n_candidates=200,
):
    """
    Query the sigmoid inside [x_lo, x_hi] until max σ(x) ≤ sigma_thresh.

    Parameters
    ----------
    x_lo, x_hi    : float  – target interval bounds
    n_initial     : int    – number of random seed observations
    sigma_thresh  : float  – stopping criterion (max posterior std)
    max_queries   : int    – hard cap on number of queries
    noise_std     : float  – observation noise std
    n_candidates  : int    – resolution of the candidate / evaluation grid

    Returns
    -------
    history : list of dict
        keys: 'X_train', 'y_train', 'mu', 'sigma', 'x_next',
              'max_sigma_interval'  (scalar: current max σ over interval)
    X_interval : ndarray, shape (n_candidates, 1)
        The dense evaluation grid restricted to [x_lo, x_hi].
    converged : bool
        True if max σ dropped below sigma_thresh before max_queries.
    """
    X_interval = np.linspace(x_lo, x_hi, n_candidates).reshape(-1, 1)

    # Initial seed observations inside the interval
    X_train = rng.uniform(x_lo, x_hi, size=(n_initial, 1))
    y_train = sigmoid_1d(X_train).ravel() + rng.randn(n_initial) * noise_std

    history = []
    converged = False

    for step in range(max_queries):

        # --- TODO 8 ----------------------------------------------------
        # 1. Build kernel + GP (same style as before), fit to X_train/y_train.
        # 2. Call max_sigma_acquisition(gp, X_interval) → (x_next, mu, sigma).
        # 3. Compute max_sigma = float(sigma.max()).
        # 4. Save a history snapshot BEFORE deciding to stop:
        #      snap = {
        #          'X_train': X_train.copy(),
        #          'y_train': y_train.copy(),
        #          'mu': mu.copy(),
        #          'sigma': sigma.copy(),
        #          'max_sigma_interval': max_sigma,
        #          'x_next': x_next.copy(),
        #      }
        #      history.append(snap)
        # 5. Stopping check: if max_sigma <= sigma_thresh,
        #    set converged = True and break.
        # 6. Observe y_next = sigmoid_1d(x_next) + noise and append to X_train/y_train.
        #
        # Tip: use a simple RBF(length_scale=1.0) with alpha=0.01 for faster,
        # more stable convergence (fewer hyperparameters to optimize).
        #
        # WRITE YOUR CODE BELOW (~15-18 lines)
        raise NotImplementedError("TODO 8: function estimation loop body")

    return history, X_interval, converged


def plot_function_estimation_1d(history, X_interval, converged, sigma_thresh=0.01):
    """
    Four-panel figure:
      Panel 1   – Convergence curve: max σ(x) over the interval vs. query index.
      Panels 2–4 – GP posterior snapshots at three stages: early, middle, final.
    """
    # --- TODO 9 --------------------------------------------------------
    #
    # Use:  fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    #
    # Panel 1 (axes[0]) – convergence curve
    #   • x-axis: query index 0 … len(history)-1
    #   • y-axis: [h['max_sigma_interval'] for h in history]
    #   • horizontal dashed red line at y = sigma_thresh, labelled 'threshold'
    #   • if converged, add a green vertical dashed line at the last step index
    #   • x-axis label: "Query index"
    #   • y-axis label: "max σ(x) over interval"
    #   • title: "Max σ(x) over interval vs. query"
    #
    # Panels 2–4 (axes[1], axes[2], axes[3]) – one snapshot each for steps:
    #     [0,  len(history) // 2,  len(history) - 1]
    #   For each snapshot panel:
    #   • ax.axvspan(x_lo, x_hi, alpha=0.10, color='gold') – shade target interval
    #   • Plot true sigmoid on a dense [-2, 2] grid as a thin black line
    #   • Plot GP mean (blue line) and ±2σ band (shaded blue) on X_interval
    #   • Scatter training points in red
    #   • If 'x_next' in h: axvline in orange
    #   • title: f"Step {idx}  (max σ = {h['max_sigma_interval']:.3f})"
    #   • Add y-axis label "f(x)" on the leftmost snapshot panel only
    #
    # WRITE YOUR CODE BELOW (~35-45 lines)
    raise NotImplementedError("TODO 9: plot convergence curve + snapshots")

    plt.tight_layout()
    plt.savefig("part2b_estimation_1d.png", dpi=150)
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

    # --- TODO 10 -------------------------------------------------------
    # Generate n_initial random 2-D points in [-3, 3]² and observe the
    # target function (with noise).
    #
    # WRITE YOUR CODE BELOW (3-4 lines)
    # X_train = ???
    # y_train = ???
    raise NotImplementedError("TODO 10: initial 2-D observations")

    history = []

    for step in range(n_queries):

        # --- TODO 11 ---------------------------------------------------
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
        raise NotImplementedError("TODO 11: active learning 2-D loop body")

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
# PART 3b — When does kernel learning fail?
# ═══════════════════════════════════════════════════════════════════════════
#
# In Part 3 you fixed the ARD length scales to domain knowledge (σ_x, σ_y)
# because freeing them caused the optimizer to produce stripes.  This part
# reveals *why* that happens and shows how to fix it with data instead.
#
# Background (Murphy §15.3.2 / Bishop §6.4.3):
#   Maximising the log marginal likelihood ("Type II ML" / empirical Bayes)
#   to learn kernel hyperparameters works well when the data is informative
#   about every dimension.  With only a handful of points in [-3,3]² the
#   likelihood surface is nearly flat along an under-sampled axis: the GP
#   can explain the observations equally well whether that axis has a short
#   or a very long length scale.  The optimiser drifts toward the upper
#   bound and the posterior uncertainty becomes stripe-shaped (all variation
#   along one axis only).
#
#   Two fixes:
#     (a) Fix length scales to prior knowledge — done in Part 3.
#     (b) Provide enough initial data that both axes are constrained.
#   Part 3b demonstrates (b) and lets you see the transition.
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
    """Run active learning on the 2-D target with a freely-optimised ARD Matérn.

    Unlike Part 3, the GP length scales are NOT fixed — they are learned by
    maximising the log marginal likelihood.  Call this function twice with
    n_initial=5 (sparse, collinear) and n_initial=25 (dense, random) to see
    how initial data coverage affects whether the optimiser recovers the
    correct anisotropy.

    Parameters
    ----------
    n_initial : int
        Number of initial observations before the active loop starts.
    n_queries : int
        Number of active-learning queries.
    X_initial : ndarray of shape (n, 2), optional
        Override the randomly generated initial design.  Supply points
        confined to a line (all y ≈0) for the sparse failure-mode run:
        the GP then has no information about y-variation, so the optimiser
        pushes ℓ_y to its upper bound, producing reliable horizontal stripes.
        If None, n_initial points are drawn uniformly from [-3, 3]².

    Returns
    -------
    history : list of dict
        Each entry has keys: 'X_train', 'y_train', 'mu_grid', 'sigma_grid',
        'x_next', 'length_scales' (the optimised ARD length scales).
    Xg, Yg : ndarray
        Meshgrid for plotting.
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
        # --- TODO 12 ---------------------------------------------------
        # Same structure as TODO 11, but use a *freely-optimised* ARD
        # Matérn kernel.  Do NOT pass optimizer=None — the whole point is
        # to let the marginal likelihood choose the length scales.
        #
        #   kernel = C(1.0, (1e-2, 1e2)) * Matern(
        #       length_scale=[1.0, 1.0],
        #       length_scale_bounds=(0.1, 20.0),
        #       nu=2.5,
        #   )
        #   gp = GaussianProcessRegressor(
        #       kernel=kernel, alpha=noise_std**2,
        #       n_restarts_optimizer=3, random_state=RNG_SEED,
        #   )
        #
        # After fitting, extract the optimised per-axis length scales:
        #   ls = gp.kernel_.k2.length_scale   # shape (2,)
        # and include 'length_scales': ls.copy() in the history snapshot
        # so the plot function can annotate each panel.
        #
        # You will likely see ConvergenceWarnings when n_initial is small —
        # that is expected and is precisely the failure mode being studied.
        #
        # WRITE YOUR CODE BELOW (~20-25 lines)
        raise NotImplementedError("TODO 12: kernel_sensitivity_2d loop body")

    return history, Xg, Yg


def plot_kernel_sensitivity_2d(
    history_sparse,
    history_dense,
    Xg,
    Yg,
    n_initial_sparse=5,
    n_initial_dense=25,
):
    """Compare uncertainty maps from sparse vs dense initial observations.

    Produces a 2 × 4 figure:
      • Top row:    history_sparse  (small n_initial — stripes expected)
      • Bottom row: history_dense   (large n_initial — ovals expected)
      • 4 columns:  steps 0, 5, 10, final

    Each panel shows the uncertainty map σ(x) and its title includes the
    optimised length scales ℓ_x and ℓ_y so you can watch how they evolve.

    Parameters
    ----------
    history_sparse, history_dense : list of dict
        As returned by kernel_sensitivity_2d.
    Xg, Yg : ndarray
        Meshgrid for plotting.
    n_initial_sparse, n_initial_dense : int
        Used for axis labels.
    """
    # --- TODO 13 -------------------------------------------------------
    # Create a 2 × 4 figure with snapshot columns at steps 0, 5, 10 and
    # the final step.  For each panel:
    #
    #   hidx = min(idx, len(history) - 1)   # guard against shorter runs
    #   ax.contourf(Xg, Yg, h['sigma_grid'], levels=20, cmap='magma')
    #   ax.scatter(h['X_train'][:, 0], h['X_train'][:, 1],
    #              c='cyan', s=15, edgecolors='k', lw=0.4)
    #   ls = h['length_scales']
    #   ax.set_title(f"step {hidx}\nℓ_x={ls[0]:.2f}, ℓ_y={ls[1]:.2f}",
    #               fontsize=8)
    #   ax.set_aspect('equal')
    #
    # Label the left y-axis of each row with n_initial and 'σ(x)'.
    # Add a suptitle summarising the lesson.
    # Save to 'part3b_kernel_sensitivity.png'.
    #
    # WRITE YOUR CODE BELOW (~25-30 lines)
    raise NotImplementedError("TODO 13: plot_kernel_sensitivity_2d")


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

    # --- TODO 14 -------------------------------------------------------
    # Implement the `init` function for FuncAnimation.
    #   • Clear both axes.
    #   • Return an empty list of artists.
    #
    # WRITE YOUR CODE BELOW (~4 lines)
    def init():
        raise NotImplementedError("TODO 14: animation init")

    # --- TODO 15 -------------------------------------------------------
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
        raise NotImplementedError("TODO 15: animation update function")

    # --- TODO 16 -------------------------------------------------------
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


raise NotImplementedError("TODO 16: create and display animation")


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
    # Initial points all near y=0: GP has no y-variation info → ℓ_y → upper bound
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
