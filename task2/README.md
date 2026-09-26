# Task 2

The quantum-model implementation is in **[task_2.py](task_2.py)**. It solves
`u''(x) + π²u(x) = 0` on `(0, 1)` with `u(0) = u(1) = 0`.
The additional condition `u(0.5) = 1` selects the nonzero solution `u(x) = sin(πx)`.

1. **Circuit:** encode x using RY(πx), apply trainable RZ/RY rotations, and
   measure the expectation of Pauli Z. The default uses one qubit and one layer;
   neighboring CNOT gates are included when more qubits are configured.
2. **Prediction:** multiply the expectation by a trainable scale and add a bias
   to obtain `u(x)`.
3. **Derivatives:** compute `u'(x)` and `u''(x)` using automatic differentiation.
4. **Training:** use Adam for 200 steps on 32 interior collocation points.
   The loss combines the mean squared equation residual with boundary and
   amplitude penalties, each weighted by 10.
5. **Results:** plot the training loss and compare the solution and both
   derivatives with their analytical expressions at the collocation points.

The qubit count, layer count, training settings, and penalty weights can be
changed in the configuration block at the top of the script.

## Install and run

From the `task2` directory, install the dependencies once:

```bash
python3 -m pip install -r requirements.txt
```

Run the program:

```bash
python3 task_2.py
```

The figures are saved in `results/training.png` and `results/derivative.png`.
