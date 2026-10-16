# E1: how low-rank are interval products of the layer transfer maps T_m = diag(Phi(t_m)) W_m ?
import numpy as np
from scipy.stats import norm
rng=np.random.default_rng(0)
def closure_T(n,L):
    Ws=[rng.normal(0,np.sqrt(2/n),(n,n)) for _ in range(L)]
    mu=np.zeros(n); S=np.eye(n); Ts=[]
    for W in Ws:
        m=W@mu; C=W@S@W.T; s=np.sqrt(np.diag(C)); t=m/s; P,p=norm.cdf(t),norm.pdf(t)
        Ts.append(P[:,None]*W)                       # T = diag(Phi(t)) W  (leg transport matrix)
        mu=m*P+s*p; e2=(m**2+s**2)*P+m*s*p; S=np.outer(P,P)*C; np.fill_diagonal(S,e2-mu**2)
    return Ts
for n,L in [(256,16),(1024,16)]:
    Ts=closure_T(n,L); M=np.eye(n); print(f"\nn={n} L={L}: rank needed for relative Frobenius error eps (interval product of 'age' layers)")
    print(" age | eps=1%  10%  30% | Gaussian-product heuristic n(1-eps^(2/age)) for 1% 10% 30%")
    for age in range(1,L):
        M=Ts[L-age]@M                               # product T_{L-1} ... T_{L-age}
        sv=np.linalg.svd(M,compute_uv=False); tail=np.sqrt(np.cumsum(sv[::-1]**2)[::-1]/np.sum(sv**2))
        ranks=[int(np.searchsorted(-tail,-e)) for e in (0.01,0.1,0.3)]
        heur=[int(n*(1-e**(2/age))) for e in (0.01,0.1,0.3)]
        if age in (1,2,3,4,6,8,10,12,15): print(f" {age:3d} | {ranks[0]:5d} {ranks[1]:4d} {ranks[2]:4d} | {heur[0]:5d} {heur[1]:4d} {heur[2]:4d}")
