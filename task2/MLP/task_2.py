"""Train a physics-informed classical MLP for u''(x) + pi^2 u(x) = 0."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a desktop window.

import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import optax

# Numerical precision.
jax.config.update("jax_enable_x64", True)


# Experiment configuration.
LAYER_SIZES = (1, 16, 1)  # Input, hidden layer 1, output.
N_COLLOCATION = 32
N_STEPS = 1000
LEARNING_RATE = 0.03
BOUNDARY_WEIGHT = 100.0
AMPLITUDE_WEIGHT = 100.0
SEED = 7


def initialize_weights(seed):
    """Create weights and biases explicitly for each layer."""
    n_input, n_hidden1, n_output = LAYER_SIZES
    key1, key2 = jax.random.split(jax.random.key(seed), 2)

    # Xavier initialization: random weights in [-limit, limit].
    # Input -> hidden layer 1 (default shapes: W1 = (1, 16), b1 = (16,)).
    limit1 = np.sqrt(6.0 / (n_input + n_hidden1))
    W1 = jax.random.uniform(
        key1, (n_input, n_hidden1), minval=-limit1, maxval=limit1,
        dtype=np.float64,
    )
    b1 = np.zeros(n_hidden1)
    
    # Hidden layer 1 -> output (W2 = (16, 1), b2 = (1,)).
    limit2 = np.sqrt(6.0 / (n_hidden1 + n_output))
    W2 = jax.random.uniform(
        key2, (n_hidden1, n_output), minval=-limit2, maxval=limit2,
        dtype=np.float64,
    )
    b2 = np.zeros(n_output)

    return [(W1, b1), (W2, b2)]
    


def mlp(x, weights):
    """Pass the input through each layer in order."""
    (W1, b1), (W2, b2)= weights

    # Give each x one input feature: a scalar becomes (1,), N points (N, 1).
    inputs = np.expand_dims(np.asarray(x), axis=-1)

    # Each hidden layer computes weighted inputs + biases, then applies activation function.
    hidden1 = jax.nn.sigmoid(inputs @ W1 + b1)

    # The output layer is tanh.
    output = np.tanh(hidden1 @ W2 + b2)

    # Return a scalar for one x, or an array of N predictions for N points.
    return np.squeeze(output, axis=-1)


def u(x, weights, readout):
    """Apply a trainable scale and bias to the MLP output."""
    scale, bias = readout
    return scale * mlp(x, weights) + bias
    #return mlp(x, weights)


def du_dx(x, weights, readout):
    """Evaluate the first input derivative at one scalar x."""
    return jax.grad(lambda z: u(z, weights, readout))(x)


def d2u_dx2(x, weights, readout):
    """Evaluate the second input derivative at one scalar x."""
    return jax.grad(lambda z: du_dx(z, weights, readout))(x)


def residual(x, weights, readout):
    """Evaluate the differential-equation residual at x."""
    return d2u_dx2(x, weights, readout) + np.pi**2 * u(x, weights, readout)


def main():
    # Create interior collocation points and initialize model parameters.
    x_collocation = (np.arange(N_COLLOCATION) + 0.5) / N_COLLOCATION
    weights = initialize_weights(SEED)
    readout = np.array([1.0, 0.0])

    def loss(weights, readout):
        """Combine equation, boundary, and amplitude penalties."""
        # Evaluate the scalar residual across all collocation points.
        residuals = jax.vmap(
            lambda x: residual(x, weights, readout)
        )(x_collocation)
        equation_loss = np.mean(residuals**2)

        boundary_loss = (
            u(0.0, weights, readout)**2
            + u(1.0, weights, readout)**2
        )

        # Fix the amplitude to select a nonzero solution.
        amplitude_loss = (u(0.5, weights, readout) - 1.0)**2

        return (
            equation_loss
            + BOUNDARY_WEIGHT * boundary_loss
            + AMPLITUDE_WEIGHT * amplitude_loss
        )

    # Train the MLP parameters and the scale/bias readout together.
    optimizer = optax.adam(learning_rate=LEARNING_RATE, b2=0.999)
    optimizer_state = optimizer.init((weights, readout))

    @jax.jit
    def train_step(weights, readout, optimizer_state):
        gradients = jax.grad(loss, argnums=(0, 1))(weights, readout)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, (weights, readout)
        )
        weights, readout = optax.apply_updates((weights, readout), updates)
        return weights, readout, optimizer_state, loss(weights, readout)

    losses = [float(loss(weights, readout))]

    for step in range(N_STEPS):
        weights, readout, optimizer_state, current_loss = train_step(
            weights, readout, optimizer_state
        )
        losses.append(float(current_loss))
        print(f"Step {step + 1}: Loss = {losses[-1]:.3e}", flush=True)

    print(f"Final training loss: {loss(weights, readout):.3e}", flush=True)

    # Evaluate input derivatives at the collocation points.
    first_derivative = np.stack(
        [du_dx(x, weights, readout) for x in x_collocation]
    )
    second_derivative = np.stack(
        [d2u_dx2(x, weights, readout) for x in x_collocation]
    )

    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)

    # Save the training loss and solution comparison.
    fig_training, axes_training = plt.subplots(1, 2, figsize=(10, 4))
    axes_training[0].plot(losses)
    axes_training[0].set(
        xlabel="Training step", ylabel="Loss", title="Training loss"
    )
    axes_training[1].plot(
        x_collocation,
        np.sin(np.pi * x_collocation),
        color="black",
        linewidth=1.8,
        label="sin(pi*x)",
    )
    axes_training[1].plot(
        x_collocation,
        u(x_collocation, weights, readout),
        "--",
        color="tab:orange",
        linewidth=1.8,
        label="Classical PINN solution",
    )
    axes_training[1].set(
        xlabel="x", ylabel="y", title="Solution at collocation points"
    )
    axes_training[1].legend()
    fig_training.tight_layout()
    fig_training.savefig(output / "training.png", dpi=150)
    plt.close(fig_training)

    # Save the first- and second-derivative comparisons.
    fig_derivatives, axes_derivatives = plt.subplots(1, 2, figsize=(10, 4))
    axes_derivatives[0].plot(
        x_collocation,
        np.pi * np.cos(np.pi * x_collocation),
        color="black",
        linewidth=1.8,
        label="Exact",
    )
    axes_derivatives[0].plot(
        x_collocation,
        first_derivative,
        "--",
        color="tab:orange",
        linewidth=1.8,
        label="Classical PINN",
    )
    axes_derivatives[0].set(
        xlabel="x", ylabel="u'(x)", title="First derivative"
    )
    axes_derivatives[0].legend()

    axes_derivatives[1].plot(
        x_collocation,
        -np.pi**2 * np.sin(np.pi * x_collocation),
        color="black",
        linewidth=1.8,
        label="Exact",
    )
    axes_derivatives[1].plot(
        x_collocation,
        second_derivative,
        "--",
        color="tab:orange",
        linewidth=1.8,
        label="Classical PINN",
    )
    axes_derivatives[1].set(
        xlabel="x", ylabel="u''(x)", title="Second derivative"
    )
    axes_derivatives[1].legend()
    fig_derivatives.tight_layout()
    fig_derivatives.savefig(output / "derivative.png", dpi=150)
    plt.close(fig_derivatives)


if __name__ == "__main__":
    main()
