# e2_conditional

- script: `experiments/conditional.py` (verbatim from the brief; seeds hard-coded in the script)
- commit: `5823c90b44cb5523d245a1e8a4716d931979d20e` (dirty); versions: {'numpy': '2.4.6', 'scipy': '1.18.1', 'flopscope': '0.12.3', 'whestbench': '0.16.1', 'opt-einsum': '3.4.0', 'threadpoolctl': '3.7.0'}
- host: vm (4 cpus), 2026-10-16T13:32:50Z, runtime 72 s, exit 0

## stdout

```
MC done in 35s; MC noise floor per neuron ~ 1.2e-07
Gaussian closure (propagated)        MSE = 3.563e-05
Gaussian closure w/ oracle mu,C_(L-1) MSE = 4.692e-06
fraction of squared error (and of output variance) in top-k eigenmodes of Cov(h_L):
   k=  1: error 0.62   variance 0.10
   k=  4: error 0.66   variance 0.29
   k= 16: error 0.75   variance 0.63
   k= 64: error 0.91   variance 0.91
   k=128: error 0.96   variance 0.97
oracle conditional closure: condition on top-k modes of h_(L-1), bath Gaussian  (k=n -> plain MC on 200k samples)
   k=  0: MSE = 4.692e-06
   k=  1: MSE = 4.242e-06
   k=  4: MSE = 3.753e-06
   k= 16: MSE = 2.160e-06
   k= 64: MSE = 1.050e-06
   k=128: MSE = 7.546e-07
   k=256: MSE = 7.857e-07
```
