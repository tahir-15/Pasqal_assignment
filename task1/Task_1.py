"""Task 1: a small quantum regressor, gradient check, and training plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save plots without needing a desktop window.

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np


# Circuit configuration.
N_QUBITS = 3
N_LAYERS = 3

device = qml.device("default.qubit", wires=N_QUBITS)


# Quantum model and target function.
@qml.qnode(device, interface="autograd", diff_method="backprop")
def circuit(x, weights):
    for qubit in range(N_QUBITS):
        qml.RY(np.pi * x, wires=qubit)  # Encode the input.

    for layer in range(N_LAYERS):
        for qubit in range(N_QUBITS):
            qml.RZ(weights[layer, qubit, 0], wires=qubit)
            qml.RY(weights[layer, qubit, 1], wires=qubit)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])

    return qml.expval(qml.PauliZ(0))


def predict(x, weights, readout):
    scale, bias = readout
    return scale * circuit(x, weights) + bias


def target(x):
    return np.sin(np.pi * x) + 0.2 * np.sin(3 * np.pi * x)


# Check parameter gradients using central finite differences.
def check_gradient(function, parameters):
    automatic = qml.grad(function)(parameters)
    numerical = np.zeros_like(parameters)
    h = 1e-6

    for index in np.ndindex(parameters.shape):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += h
        minus[index] -= h
        numerical[index] = (function(plus) - function(minus)) / (2 * h)

    assert np.allclose(
        automatic, numerical, atol=1e-7, rtol=1e-5
    ), "Gradient check failed"

    return float(np.max(np.abs(automatic - numerical)))


def main():
    # Create training data and a separate evaluation grid.
    x_train = np.array(np.linspace(-0.98, 0.98, 64), requires_grad=False)
    y_train = target(x_train)

    # These midpoint samples are never used by the optimizer.
    x_test = np.array(
        -1 + 2 * (np.arange(501) + 0.5) / 501,
        requires_grad=False,
    )

    # Initialize the parameters and record the untrained prediction.
    np.random.seed(7)
    weights = np.array(
        np.random.uniform(-np.pi, np.pi, (N_LAYERS, N_QUBITS, 2)),
        requires_grad=True,
    )
    readout = np.array([1.0, 0.0], requires_grad=True)
    prediction_before = predict(x_test, weights, readout)

    def loss(weights, readout):
        return np.mean((predict(x_train, weights, readout) - y_train) ** 2)

    # Verify gradients before training.
    weight_error = check_gradient(lambda w: loss(w, readout), weights)
    readout_error = check_gradient(lambda r: loss(weights, r), readout)
    print(
        f"Gradient check passed. Maximum error: "
        f"{max(weight_error, readout_error):.2e}",
        flush=True,
    )

    # Train with Adam and record the loss after each update.
    optimizer = qml.AdamOptimizer(stepsize=0.03, beta2=0.999)
    losses = [float(loss(weights, readout))]

    for step in range(500):
        weights, readout = optimizer.step(loss, weights, readout)
        losses.append(float(loss(weights, readout)))

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}: MSE = {losses[-1]:.3e}", flush=True)

    # Evaluate the trained model on unseen inputs.
    prediction = predict(x_test, weights, readout)
    test_mse = float(np.mean((prediction - target(x_test)) ** 2))
    print(f"Held-out MSE: {test_mse:.3e}", flush=True)
    assert test_mse < 1e-4, "The model did not learn the target accurately."

    # Prepare the output directory and comparison figure.
    output = Path(__file__).parent / "results"
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: training loss.
    axes[0].plot(losses)
    axes[0].set(
        xlabel="Training step",
        ylabel="Mean squared error",
        title="Training loss",
    )

    # Right panel: target and predictions before and after training.
    axes[1].plot(
        x_test,
        target(x_test),
        color="black",
        linewidth=3,
        label="Target",
    )
    axes[1].plot(
        x_test,
        prediction_before,
        ":",
        color="tab:blue",
        linewidth=2,
        label="Before training",
    )
    axes[1].plot(
        x_test,
        prediction,
        "--",
        color="tab:orange",
        linewidth=1.8,
        label="After training",
    )
    axes[1].set(xlabel="x", ylabel="y", title="Fit on unseen inputs")
    axes[1].legend()

    # Save the figure, trained parameters, and numerical summary.
    fig.tight_layout()
    fig.savefig(output / "training.png", dpi=150)
    plt.close(fig)

    np.savez(output / "model.npz", weights=weights, readout=readout)
    (output / "training_summary.txt").write_text(
        f"Gradient maximum error: {max(weight_error, readout_error):.6e}\n"
        f"Initial training MSE: {losses[0]:.6e}\n"
        f"Final training MSE: {losses[-1]:.6e}\n"
        f"Held-out MSE: {test_mse:.6e}\n"
    )
    print(f"Plots saved to {output / 'training.png'}")


if __name__ == "__main__":
    main()
