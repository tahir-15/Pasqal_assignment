# Task 3: Physics-informed MLP

[task_3.py](task_3.py) uses JAX to train a classical neural network for a family
of damped oscillators:

```text
x''(t) + 2 gamma x'(t) + omega² x(t) = 0
x(0) = 1, x'(0) = 0
```

The inputs are time `t`, damping `gamma`, and frequency `omega`. The output
is the displacement `x(t)`.

## Method

1. Generate numerical references with SciPy's `solve_ivp` using RK45.
   Training uses 20 damping values in `[0, 1]`, 20 frequency values in `[1, 4]`,
   and 101 time points in `[0, 10]` for each of the 400 parameter pairs.
2. Use two hidden layers of 32 neurons each, with sine and cosine activations,
   followed by a single output with a `tanh` activation.
3. Compute time derivatives with JAX and train with a combined loss:
   `data_loss + 0.1 * physics_loss + 0.1 * initial_loss`.
   Optax Adam runs for 5000 steps with a learning rate of 0.03.
4. Compare predictions with the numerical reference at the unseen parameter
   pair `gamma = 0.5`, `omega = 2.5`, and print the root mean square error (RMSE).

## Install and run

From the `task3` folder, install the dependencies:

```bash
python3 -m pip install -r requirement.txt
```

Then run:

```bash
python3 task_3.py
```



## Output

The script prints the training loss and unseen-pair RMSE, and saves:

- `results/training_loss.png`: total loss against optimization iteration.
- `results/prediction_comparison.png`: MLP prediction and numerical reference
  for the unseen parameter pair.
