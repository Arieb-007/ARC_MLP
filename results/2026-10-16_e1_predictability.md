# e1_predictability

- script: `experiments/predictability.py` (verbatim from the brief; seeds hard-coded in the script)
- commit: `5823c90b44cb5523d245a1e8a4716d931979d20e` (dirty); versions: {'numpy': '2.4.6', 'scipy': '1.18.1', 'flopscope': '0.12.3', 'whestbench': '0.16.1', 'opt-einsum': '3.4.0', 'threadpoolctl': '3.7.0'}
- host: vm (4 cpus), 2026-10-16T13:31:38Z, runtime 6 s, exit 0

## stdout

```

n=256 L=16: rank needed for relative Frobenius error eps (interval product of 'age' layers)
 age | eps=1%  10%  30% | Gaussian-product heuristic n(1-eps^(2/age)) for 1% 10% 30%
   1 |   173  127   87 |   255  253  232
   2 |   141   95   57 |   253  230  179
   3 |   122   76   43 |   244  200  141
   4 |   109   64   35 |   230  175  115
   6 |    90   50   27 |   200  137   84
   8 |    76   41   21 |   175  112   66
  10 |    66   34   17 |   154   94   54
  12 |    58   29   15 |   137   81   46
  15 |    51   25   13 |   117   67   37

n=1024 L=16: rank needed for relative Frobenius error eps (interval product of 'age' layers)
 age | eps=1%  10%  30% | Gaussian-product heuristic n(1-eps^(2/age)) for 1% 10% 30%
   1 |   656  507  351 |  1023 1013  931
   2 |   563  388  236 |  1013  921  716
   3 |   491  312  179 |   976  803  565
   4 |   437  264  145 |   921  700  463
   6 |   354  198  104 |   803  548  338
   8 |   300  160   80 |   700  448  266
  10 |   260  133   66 |   616  377  219
  12 |   230  115   56 |   548  326  186
  15 |   199   98   47 |   469  270  151
```
