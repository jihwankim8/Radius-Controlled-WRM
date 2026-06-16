# Radius-Controlled WRM on MNIST

This repository contains a minimal experiment for Radius-Controlled Wasserstein Robust Method (RC-WRM) on MNIST.

The experiment compares Radius-Controlled WRM with fixed-gamma WRM baselines and saves the resulting effective-radius and approximate Wasserstein worst-case loss plots.

## Files

```text
mnist_RC-WRM.py
run_mnist_RC-WRM.sh
```

## Requirements

```bash
pip install torch torchvision numpy matplotlib
```

## Run

```bash
bash run_mnist_RC-WRM.sh
```

## Outputs

The script saves results under:

```text
runs/mnist_RC-WRM/
```

Main outputs:

```text
effective_radius.png
wasserstein_worstcase_loss.png
wall_clock.txt
wall_clock.csv
```

## Experiment Summary

The Radius-Controlled WRM run updates the penalty parameter gamma so that the effective radius approaches the target radius rho0. Fixed-gamma WRM baselines are also trained for comparison.

Default settings:

```text
rho0 = 0.3
gamma_init = 0.4
fixed gammas = 0.2, 0.4, 0.6, 0.8, 1.0
epochs = 4
train samples = 12000
test samples = 2000
```
