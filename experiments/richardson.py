# E3: is the closure error (K=2) predictable from the K=1 -> K=2 difference (Richardson-type extrapolation)?
# and is the dominant error direction the radial/mean direction?
import numpy as np
from scipy.stats import norm
psi=lambda m,s: m*norm.cdf(m/s)+s*norm.pdf(m/s)
def run(seed,n=256,L=8,N=800_000,B=100_000):
    rng=np.random.default_rng(seed); Ws=[rng.normal(0,np.sqrt(2/n),(n,n)) for _ in range(L)]
    mu=np.zeros(n); S=np.eye(n)
    for W in Ws:
        m=W@mu; C=W@S@W.T; s=np.sqrt(np.diag(C)); P,p=norm.cdf(m/s),norm.pdf(m/s)
        mu=m*P+s*p; e2=(m**2+s**2)*P+m*s*p; S=np.outer(P,P)*C; np.fill_diagonal(S,e2-mu**2)
    mu2=mu
    mu=np.zeros(n); tr=float(n)
    for W in Ws:
        m=W@mu; s=np.sqrt((W**2).sum(1)*tr/n); P,p=norm.cdf(m/s),norm.pdf(m/s)
        mu=m*P+s*p; e2=(m**2+s**2)*P+m*s*p; tr=float((e2-mu**2).sum())
    mu1=mu
    sumL=np.zeros(n); sumLL=np.zeros((n,n))
    for b in range(N//B):
        h=rng.normal(size=(B,n))
        for W in Ws: h=np.maximum(h@W.T,0)
        sumL+=h.sum(0); sumLL+=h.T@h
    mt=sumL/N; CL=sumLL/N-np.outer(mt,mt); lam,U=np.linalg.eigh(CL); u1=U[:,-1]
    e2_=mu2-mt; d21=mu2-mu1
    alpha=-(e2_@d21)/(d21@d21)
    print(f"seed {seed}: MSE K=1 {np.mean((mu1-mt)**2):.2e}  K=2 {np.mean(e2_**2):.2e} | corr(err_K2, K2-K1) = {np.corrcoef(e2_,d21)[0,1]:+.3f}, "
          f"oracle alpha={alpha:+.3f} -> MSE {np.mean((mu2+alpha*d21-mt)**2):.2e} | cos(u1,mu)={abs(u1@mt)/np.linalg.norm(mt):.3f}, "
          f"cos(err,mu)={abs(e2_@mt)/np.linalg.norm(mt)/np.linalg.norm(e2_):.3f}")
    return alpha
alphas=[run(s) for s in (3,4,5)]
