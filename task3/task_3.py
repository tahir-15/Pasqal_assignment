"""Train a physics-informed MLP for a family of damped oscillators."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a desktop window.

import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import optax
from scipy.integrate import solve_ivp

# Numerical precision.
jax.config.update("jax_enable_x64", True)

# Model and training configuration.
LAYER_SIZES = (3, 32, 32, 1)  # Input, two hidden layers, output.
SEED = 7
PHYSICS_WEIGHT = 0.1
INITIAL_WEIGHT = 0.1
LEARNING_RATE = 0.03
BETA1 = 0.9    # Moving average of gradients; use 0 <= BETA1 < 1.
BETA2 = 0.999  # Moving average of squared gradients; use 0 <= BETA2 < 1.
N_STEPS = 5000


def initialize_weights(seed):
    """Initialize each layer with Xavier uniform weights and zero biases."""
    n_input, n_hidden1, n_hidden2, n_output = LAYER_SIZES
    key1, key2, key3 = jax.random.split(jax.random.key(seed), 3)

    # Input -> first hidden layer: W1 is (3, 32), b1 is (32,).
    limit1 = np.sqrt(6.0 / (n_input + n_hidden1))
    W1 = jax.random.uniform(
        key1,
        (n_input, n_hidden1),
        minval=-limit1,
        maxval=limit1,
        dtype=np.float64,
    )
    b1 = np.zeros(n_hidden1)

    # First -> second hidden layer: W2 is (32, 32), b2 is (32,).
    limit2 = np.sqrt(6.0 / (n_hidden1 + n_hidden2))
    W2 = jax.random.uniform(
        key2,
        (n_hidden1, n_hidden2),
        minval=-limit2,
        maxval=limit2,
        dtype=np.float64,
    )
    b2 = np.zeros(n_hidden2)

    # Second hidden layer -> output: W3 is (32, 1), b3 is (1,).
    limit3 = np.sqrt(6.0 / (n_hidden2 + n_output))
    W3 = jax.random.uniform(
        key3,
        (n_hidden2, n_output),
        minval=-limit3,
        maxval=limit3,
        dtype=np.float64,
    )
    b3 = np.zeros(n_output)
    return [(W1, b1), (W2, b2), (W3, b3)]


def mlp(inputs, weights):
    """Map [time, damping, frequency] inputs to displacement predictions."""
    (W1, b1), (W2, b2), (W3, b3) = weights

    hidden1 = np.sin(inputs @ W1 + b1)
    hidden2 = np.cos(hidden1 @ W2 + b2)
    output = np.tanh(hidden2 @ W3 + b3)
    return np.squeeze(output, axis=-1)


def predict(t, gamma, omega, weights, readout):
    """Predict one displacement; the readout argument is currently inactive."""
    inputs = np.array([t, gamma, omega])
    scale, bias = readout
    return mlp(inputs, weights)


def dx_dt(t, gamma, omega, weights, readout):
    """Differentiate the predicted displacement with respect to time."""
    return jax.grad(
        lambda time: predict(time, gamma, omega, weights, readout)
    )(t)


def d2x_dt2(t, gamma, omega, weights, readout):
    """Differentiate the predicted velocity with respect to time."""
    return jax.grad(
        lambda time: dx_dt(time, gamma, omega, weights, readout)
    )(t)


def residual(t, gamma, omega, weights, readout):
    """Evaluate the damped-oscillator differential equation residual."""
    x = predict(t, gamma, omega, weights, readout)
    velocity = dx_dt(t, gamma, omega, weights, readout)
    acceleration = d2x_dt2(t, gamma, omega, weights, readout)
    return acceleration + 2 * gamma * velocity + omega**2 * x


def oscillator(t, state, gamma, omega):
    """Express the oscillator equation as displacement and velocity rates."""
    x, v = state
    dx_dt = v
    dv_dt = -2 * gamma * v - omega**2 * x
    return [dx_dt, dv_dt]


def reference_solution(times, gamma, omega):
    """Solve the oscillator numerically with initial displacement 1 and velocity 0."""
    solution = solve_ivp(
        oscillator,
        t_span=(float(times[0]), float(times[-1])),
        y0=[1.0, 0.0],
        args=(gamma, omega),
        t_eval=times,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return solution.y[0]


def main():
    """Prepare reference data, train the model, and save comparison plots."""
    # Generate numerical reference trajectories.
    times = np.linspace(0.0, 10.0, 101)
    gamma_values = np.linspace(0.0, 1.0, 20)
    omega_values = np.linspace(1.0, 4.0, 20)

    parameter_pairs = []
    trajectories = []

    for gamma in gamma_values:
        for omega in omega_values:
            x_reference = reference_solution(times, gamma, omega)
            parameter_pairs.append([gamma, omega])
            trajectories.append(x_reference)

    parameter_pairs = np.array(parameter_pairs)
    trajectories = np.stack(trajectories)

    # Repeat the time grid for each oscillator.
    t_train = np.tile(times, len(parameter_pairs))

    # Repeat each oscillator's parameters for all its time points.
    gamma_train = np.repeat(parameter_pairs[:, 0], len(times))
    omega_train = np.repeat(parameter_pairs[:, 1], len(times))

    # Each row contains [time, damping, frequency].
    inputs_train = np.column_stack([t_train, gamma_train, omega_train])

    # Flatten the trajectories in the same oscillator-by-oscillator order.
    targets_train = trajectories.reshape(-1)

    # Initialize the model parameters.
    weights = initialize_weights(SEED)
    # Retained in the optimizer interface; readout is inactive in this model.
    readout = np.array([1.0, 0.0])

    def loss(weights, readout):
        """Combine data, differential equation, and initial-condition errors."""
        scale, bias = readout

        # Match the reference displacement values.
        predictions = mlp(inputs_train, weights)
        data_loss = np.mean((predictions - targets_train)**2)

        # Enforce the ODE at the training input points.
        residuals = jax.vmap(
            lambda t, gamma, omega: residual(
                t, gamma, omega, weights, readout
            )
        )(t_train, gamma_train, omega_train)

        physics_loss = np.mean(residuals**2)

        # Evaluate initial conditions for each parameter pair.
        initial_positions = jax.vmap(
            lambda pair: predict(0.0, pair[0], pair[1], weights, readout)
        )(parameter_pairs)

        initial_velocities = jax.vmap(
            lambda pair: dx_dt(0.0, pair[0], pair[1], weights, readout)
        )(parameter_pairs)

        initial_loss = (
            np.mean((initial_positions - 1.0)**2)
            + np.mean(initial_velocities**2)
        )

        return (
            data_loss
            + PHYSICS_WEIGHT * physics_loss
            + INITIAL_WEIGHT * initial_loss
        )

    optimizer = optax.adam(
        learning_rate=LEARNING_RATE,
        b1=BETA1,
        b2=BETA2,
    )
    optimizer_state = optimizer.init((weights, readout))

    @jax.jit
    def train_step(weights, readout, optimizer_state):
        """Apply one Adam update and return the updated loss."""
        # The inactive readout has zero gradients; only the network learns.
        gradients = jax.grad(loss, argnums=(0, 1))(weights, readout)

        # Calculate and apply Adam updates.
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, (weights, readout)
        )
        weights, readout = optax.apply_updates((weights, readout), updates)

        # Evaluate the loss after the update.
        current_loss = loss(weights, readout)
        return weights, readout, optimizer_state, current_loss

    # Record the initial loss and the loss after each training step.
    losses = [float(loss(weights, readout))]

    for step in range(N_STEPS):
        weights, readout, optimizer_state, current_loss = train_step(
            weights, readout, optimizer_state
        )
        losses.append(float(current_loss))

        if (step + 1) % 2 == 0:
            print(
                f"Step {step + 1}: Loss = {losses[-1]:.6e}",
                flush=True,
            )
    print("Final training loss:", losses[-1])

    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)

    # Save the optimization history.
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set(
        xlabel="Optimization iteration",
        ylabel="Total loss",
        title="Training loss",
    )
    fig.tight_layout()
    fig.savefig(output / "training_loss.png", dpi=150)
    plt.close(fig)

    # Compare the trained model with a reference for an unseen parameter pair.
    gamma_test = 0.5
    omega_test = 2.5
    reference = reference_solution(times, gamma_test, omega_test)
    prediction = jax.vmap(
        lambda t: predict(t, gamma_test, omega_test, weights, readout)
    )(times)

    rmse = np.sqrt(np.mean((prediction - reference)**2))
    print(f"Unseen-pair RMSE: {float(rmse):.6e}")

    fig, ax = plt.subplots()
    ax.plot(times, reference, color="black", label="Numerical reference")
    ax.plot(
        times, prediction, "--", color="tab:orange", label="MLP prediction"
    )
    ax.set(
        xlabel="Time t",
        ylabel="Displacement x(t)",
        title=f"Unseen pair: gamma={gamma_test}, omega={omega_test}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "prediction_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
