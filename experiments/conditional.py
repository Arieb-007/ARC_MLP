# E2: (a) where does Gaussian-closure error live (slow modes vs bulk)?  (b) oracle test of "condition on k slow
# modes of h_{L-1}, treat the rest as Gaussian" at the final layer.
import numpy as np, time
from scipy.stats import norm
rng=np.random.default_rng(3); n,L=256,8
Ws=[rng.normal(0,np.sqrt(2/n),(n,n)) for _ in range(L)]
psi=lambda m,s: m*norm.cdf(m/s)+s*norm.pdf(m/s)
# ---- Gaussian closure (kit-style covariance propagation) ----
mu=np.zeros(n); S=np.eye(n); props=[]
for W in Ws:
    m=W@mu; C=W@S@W.T; s=np.sqrt(np.diag(C)); P,p=norm.cdf(m/s),norm.pdf(m/s)
    mu=m*P+s*p; e2=(m**2+s**2)*P+m*s*p; S=np.outer(P,P)*C; np.fill_diagonal(S,e2-mu**2); props.append((mu.copy(),S.copy()))
mu_gc=props[-1][0]
# ---- Monte Carlo ground truth + kept samples of h_{L-1} ----
N,B,keep=1_600_000,100_000,200_000
sumL=np.zeros(n); kept=[]; sumP=np.zeros(n); sumPP=np.zeros((n,n)); t0=time.time()
for b in range(N//B):
    h=rng.normal(size=(B,n))
    for W in Ws[:-1]: h=np.maximum(h@W.T,0)
    sumP+=h.sum(0); sumPP+=h.T@h
    if len(kept)*B<keep: kept.append(h.astype(np.float32))
    sumL+=np.maximum(h@Ws[-1].T,0).sum(0)
mu_true=sumL/N; muP=sumP/N; CP=sumPP/N-np.outer(muP,muP); H=np.vstack(kept).astype(np.float64)
print(f"MC done in {time.time()-t0:.0f}s; MC noise floor per neuron ~ {np.mean(np.diag(Ws[-1]@CP@Ws[-1].T))*0.5/N:.1e}")
err=lambda est: np.mean((est-mu_true)**2)
print(f"Gaussian closure (propagated)        MSE = {err(mu_gc):.3e}")
W=Ws[-1]; m=W@muP; s2full=np.diag(W@CP@W.T)
print(f"Gaussian closure w/ oracle mu,C_(L-1) MSE = {err(psi(m,np.sqrt(s2full))):.3e}")
hL=np.maximum(H@W.T,0); CL=np.cov(hL.T); lam,U=np.linalg.eigh(CL); lam,U=lam[::-1],U[:,::-1]
e=mu_gc-mu_true; proj=(U.T@e)**2; cum=np.cumsum(proj)/proj.sum(); cumvar=np.cumsum(lam)/lam.sum()
print("fraction of squared error (and of output variance) in top-k eigenmodes of Cov(h_L):")
for k in (1,4,16,64,128): print(f"   k={k:3d}: error {cum[k-1]:.2f}   variance {cumvar[k-1]:.2f}")
lamP,UP=np.linalg.eigh(CP); lamP,UP=lamP[::-1],UP[:,::-1]
A=(H-muP)@UP; WU=W@UP
print("oracle conditional closure: condition on top-k modes of h_(L-1), bath Gaussian  (k=n -> plain MC on 200k samples)")
for k in (0,1,4,16,64,128,256):
    s2=s2full-(WU[:,:k]**2*lamP[:k]).sum(1); s=np.sqrt(np.maximum(s2,1e-12))
    est=np.zeros(n)
    for c in range(0,keep,20000):
        M=m[None,:]+A[c:c+20000,:k]@WU[:,:k].T
        est+=(psi(M,s[None,:]) if k<n else np.maximum(M,0)).sum(0)
    print(f"   k={k:3d}: MSE = {err(est/keep):.3e}")
