# Task 2: Classical MLP

[task_2.py](task_2.py) uses JAX to train a classical physics-informed neural
network for `u''(x) + π²u(x) = 0` on `(0, 1)`, with `u(0) = u(1) = 0`.
The additional condition `u(0.5) = 1` selects `u(x) = sin(πx)`.


1. Here we MLP with one input, one hidden layers of
   16 neurons with `sigmoid` activation, and one  output with 'tanh' activation. Keep the trainable
   scale and bias in `u`.
2. Calculate the first and second input derivatives using nested `jax.grad`
   calls. Use `jax.vmap` to evaluate the residual at all collocation points.
3. Train with Optax Adam for 1000 steps at learning rate 0.03, using the same
   32 interior points and boundary/amplitude penalty weights of 100. JAX compiles the training step using `jax.jit`.
4. Plot the loss, solution, and both derivatives.



## Install and run

From `task2/MLP`, install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Then run:

```bash
python3 task_2.py
```

The figures are saved in `results/training.png` and `results/derivative.png`.

