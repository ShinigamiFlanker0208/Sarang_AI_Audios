import jax
import jax.numpy as jnp
from jax import Array

# We define fixed static parameters for the XLA compiler
# 44100Hz sample rate. tau_max of 1000 gives us a lowest detectable frequency of ~44Hz (Standard Guitar E2 is 82Hz).
TAU_MAX = 1000
FRAME_SIZE = 2048


@jax.jit
def compute_difference_function(audio_frame: jnp.ndarray) -> jnp.ndarray:
    """
    Step 1: The Difference Function.
    Calculates the squared difference between the signal and a delayed version of itself.
    """
    # The integration window (W)
    w = FRAME_SIZE - TAU_MAX

    # We define a single shift operation
    def compute_lag(tau):
        # We use dynamic_slice for XLA compatibility (much faster than standard indexing)
        original_window = jax.lax.dynamic_slice(audio_frame, (0,), (w,))
        shifted_window = jax.lax.dynamic_slice(audio_frame, (tau,), (w,))

        diff = original_window - shifted_window
        return jnp.sum(diff ** 2)

    # Vectorize the operation across all possible lags (0 to TAU_MAX - 1)
    taus = jnp.arange(TAU_MAX)

    # vmap transforms the single-lag function into a batched function instantly
    diffs = jax.vmap(compute_lag)(taus)
    return diffs


@jax.jit
def compute_cmnd(diffs: jnp.ndarray) -> jnp.ndarray:
    """
    Step 2: Cumulative Mean Normalized Difference.
    Normalizes the difference function to prevent choosing the zero-lag (which is always 0).
    """
    # Create an array of divisors: [1, 2, 3, ..., TAU_MAX]
    tau_range = jnp.arange(1, TAU_MAX + 1)

    # Cumulative sum of the difference function
    cumulative_sum = jnp.cumsum(diffs)

    # Calculate the running mean
    running_mean = cumulative_sum / tau_range

    # Normalize. jnp.where handles the division by zero safely for XLA.
    cmnd = jnp.where(running_mean == 0, 1.0, diffs / running_mean)

    # The YIN paper explicitly states cmnd must be 1.0
    cmnd = cmnd.at[0].set(1.0)

    return cmnd


@jax.jit
def parabolic_interpolation(cmnd: jnp.ndarray, tau: int) -> Array:
    """
    Step 4: Parabolic Interpolation.
    Finds the exact sub-sample minimum of the CMND valley.
    """
    # XLA-safe boundary checks (prevent out-of-bounds array access)
    t_prev = jnp.maximum(tau - 1, 0)
    t_next = jnp.minimum(tau + 1, TAU_MAX - 1)

    y0 = cmnd[t_prev]
    y1 = cmnd[tau]
    y2 = cmnd[t_next]

    # Parabola vertex formula: delta = (y0 - y2) / (2 * (y0 - 2*y1 + y2))
    denominator = 2.0 * (y0 - 2.0 * y1 + y2)

    # jnp.where prevents division by zero if the curve is perfectly flat
    delta = jnp.where(denominator == 0.0, 0.0, (y0 - y2) / denominator)

    return tau + delta


@jax.jit
def extract_pitch(cmnd: jnp.ndarray, sample_rate: float, threshold: float = 0.10) -> jnp.ndarray:
    """
    Absolute Thresholding & Pitch Extraction (Interpolated).
    Note: Threshold lowered to 0.10 to prevent 'Early Dip' errors on clean signals.
    """
    valid_mask = (cmnd < threshold) & (jnp.arange(TAU_MAX) > 0)

    tau_estimate = jnp.argmax(valid_mask)
    global_min_tau = jnp.argmin(cmnd[1:]) + 1

    final_tau_int = jax.lax.select(
        tau_estimate > 0,
        tau_estimate,
        global_min_tau
    )

    # Apply Parabolic Interpolation to get floating-point accuracy
    refined_tau = parabolic_interpolation(cmnd, final_tau_int)

    pitch_hz = jnp.where(refined_tau > 0.0, sample_rate / refined_tau, 0.0)
    return pitch_hz


