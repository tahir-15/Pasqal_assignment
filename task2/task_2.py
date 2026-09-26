"""Train a physics-informed quantum circuit for u''(x) + pi^2 u(x) = 0."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a desktop window.

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np


# Experiment configuration.
N_QUBITS = 1
N_LAYERS = 1
N_COLLOCATION = 32
N_STEPS = 200
LEARNING_RATE = 0.03
BOUNDARY_WEIGHT = 10.0
AMPLITUDE_WEIGHT = 10.0
SEED = 7

device = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(device, interface="autograd", diff_method="backprop")
def circuit(x, weights):
    """Encode x, apply trainable layers, and return the Z expectation."""
    for qubit in range(N_QUBITS):
        qml.RY(np.pi * x, wires=qubit)

    for layer in range(N_LAYERS):
        for qubit in range(N_QUBITS):
            qml.RZ(weights[layer, qubit, 0], wires=qubit)
            qml.RY(weights[layer, qubit, 1], wires=qubit)

        for qubit in range(N_QUBITS - 1):
            qml.CNOT(wires=[qubit, qubit + 1])

    return qml.expval(qml.PauliZ(0))


def u(x, weights, readout):
    """Apply a trainable scale and bias to the circuit output."""
    scale, bias = readout
    return scale * circuit(x, weights) + bias


def du_dx(x, weights, readout):
    """Evaluate the first input derivative at one scalar x."""
    return qml.grad(lambda z: u(z, weights, readout))(x)


def d2u_dx2(x, weights, readout):
    """Evaluate the second input derivative at one scalar x."""
    return qml.grad(lambda z: du_dx(z, weights, readout))(x)


def residual(x, weights, readout):
    """Evaluate the differential-equation residual at x."""
    return d2u_dx2(x, weights, readout) + np.pi**2 * u(x, weights, readout)


def main():
    # Create interior collocation points and initialize model parameters.
    x_collocation = np.array(
        (np.arange(N_COLLOCATION) + 0.5) / N_COLLOCATION,
        requires_grad=True,
    )
    np.random.seed(SEED)
    weights = np.array(
        np.random.uniform(-np.pi, np.pi, (N_LAYERS, N_QUBITS, 2)),
        requires_grad=True,
    )
    readout = np.array([1.0, 0.0], requires_grad=True)

    def loss(weights, readout):
        """Combine equation, boundary, and amplitude penalties."""
        residuals = np.stack(
            [residual(x, weights, readout) for x in x_collocation]
        )
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

    # Train the circuit angles and the classical readout together.
    optimizer = qml.AdamOptimizer(stepsize=LEARNING_RATE, beta2=0.999)
    losses = [float(loss(weights, readout))]

    for step in range(N_STEPS):
        weights, readout = optimizer.step(loss, weights, readout)
        losses.append(float(loss(weights, readout)))
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
        label="Quantum PINN solution",
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
        label="Quantum PINN",
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
        label="Quantum PINN",
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
