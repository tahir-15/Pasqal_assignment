# Task 1

Everything is in **Task_1.py**. Read it from top to bottom:

1. **Circuit:** encode x using RY(πx) on three qubits. Apply three layers of
   trainable RZ/RY rotations and CNOT gates. Measure the expectation of Pauli Z.
2. **Prediction:** multiply that expectation by a trainable scale and add a bias.
3. **Gradient check:** compare automatic differentiation with central finite
   differences for all 18 circuit angles and both readout parameters.
4. **Training:** minimize mean squared error using Adam for 500 steps.
5. **Results:** evaluate on unseen inputs and save the loss and prediction plots.

## Install and run

From the `task1` directory, install the dependencies once:

```bash
python3 -m pip install -r requirements.txt
```

Run the program:

```bash
python3 Task_1.py
```


