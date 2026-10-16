# e3_richardson

- script: `experiments/richardson.py` (verbatim from the brief; seeds hard-coded in the script)
- commit: `5823c90b44cb5523d245a1e8a4716d931979d20e` (dirty); versions: {'numpy': '2.4.6', 'scipy': '1.18.1', 'flopscope': '0.12.3', 'whestbench': '0.16.1', 'opt-einsum': '3.4.0', 'threadpoolctl': '3.7.0'}
- host: vm (4 cpus), 2026-10-16T13:33:42Z, runtime 52 s, exit 0

## stdout

```
seed 3: MSE K=1 8.67e-04  K=2 3.55e-05 | corr(err_K2, K2-K1) = +0.031, oracle alpha=-0.003 -> MSE 3.55e-05 | cos(u1,mu)=0.947, cos(err,mu)=0.843
seed 4: MSE K=1 1.19e-03  K=2 2.28e-05 | corr(err_K2, K2-K1) = -0.214, oracle alpha=+0.041 -> MSE 2.10e-05 | cos(u1,mu)=0.866, cos(err,mu)=0.763
seed 5: MSE K=1 1.02e-03  K=2 3.85e-05 | corr(err_K2, K2-K1) = +0.029, oracle alpha=-0.005 -> MSE 3.85e-05 | cos(u1,mu)=0.943, cos(err,mu)=0.797
```
