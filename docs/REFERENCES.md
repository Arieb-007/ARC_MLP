# References

> Compiled by an LLM agent (Claude Code). Entries marked *(not verified)* were given to us in the
> task brief and could not be opened from the sandbox (arXiv, AIcrowd forum and HF are blocked by
> its network policy); bibliographic details of classical papers are from memory of the standard
> citations and should be checked before they go into the PDF.

## Challenge

- ARC White-Box Estimation Challenge 2026 (AIcrowd), challenge page:
  https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026
- Phase-2 announcement: https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197 *(not verified)*
- Algorithmic-contribution prize guidelines: https://discourse.aicrowd.com/t/algorithmic-contribution-prize-guidelines-how-arc-judges-these-prizes-discretion-technical-writeups-llm-usage/18041 *(not verified)*
- Starter kit: https://github.com/AIcrowd/whest-starterkit (read: `docs/reference/estimator-contract.md`,
  `docs/reference/rounds.md`, `docs/how-to/use-evaluation-datasets.md`,
  `docs/getting-started/stage-5-package.md`)
- whestbench: https://github.com/AIcrowd/whestbench (PyPI `whestbench` 0.16.1 used here)
- flopscope: https://github.com/AIcrowd/flopscope (PyPI `flopscope` 0.12.3 used here)
- Dataset: https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026, revision `v2-phase2`
- Contact for rules questions: arc-whestbench@aicrowd.com

## Prior art in the competition

- Companion paper: Wu, Lecomte, Winer, Robinson, Hilton, Christiano, arXiv:2605.05179 (cumulant
  propagation; factorized K=3, §4.3) *(not verified — arXiv blocked; cited as given)*
- ARC reference implementation of cumulant propagation (MIT), cited by 504aldo:
  https://github.com/alignment-research-center/mlp_cumulant_propagation *(not opened)*
- 504aldo, "Everything we tried: a factorized K=3 cumulant-propagation estimator at 0.25 x B …",
  forum topic 18218:
  https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218
  — full text mirrored in https://github.com/504aldo/whest-p2-cumulant-k3 (`docs/community_post.md`,
  `docs/findings_log.md` incl. F105 correction), commit `1a4083f` (read).
- V29 source (MIT): https://github.com/504aldo/whest-p2-cumulant-k3/blob/main/estimators/estimator_v29.py
  (vendored in `third_party/v29/`).
- Hisernberg's submission archive incl. V29-verbatim grader report #333467:
  https://github.com/Hisernberg/arc-whitebox (commit `b72d12f`, read).

## Mathematical background

- L. Isserlis (1918). On a formula for the product-moment coefficient of any order of a normal
  frequency distribution in any number of variables. *Biometrika* 12(1/2), 134–139.
- C. M. Stein (1981). Estimation of the mean of a multivariate normal distribution. *Annals of
  Statistics* 9(6), 1135–1151. (Stein's lemma `E[x·∇u] = E[Δu]` used in F7.)
- R. Price (1958). A useful theorem for nonlinear devices having Gaussian inputs. *IRE Trans.
  Information Theory* 4(2), 69–72. (Price's theorem, F1.)
- C. M. Newman (1986). The distribution of Lyapunov exponents: exact results for random matrices.
  *Communications in Mathematical Physics* 103(1), 121–126. (Lyapunov spectrum of Gaussian
  products, F4.)
- B. C. Moore (1981). Principal component analysis in linear systems: controllability,
  observability, and model reduction. *IEEE Trans. Automatic Control* 26(1), 17–32. (Balanced
  truncation / Gramians, Stage 2.4.)
- C. J. Oates, M. Girolami, N. Chopin (2017). Control functionals for Monte Carlo integration.
  *JRSS B* 79(3), 695–718. (Stein control variates, F7.)
- D. A. Roberts, S. Yaida, B. Hanin (2022). *The Principles of Deep Learning Theory*. Cambridge
  University Press; arXiv:2106.10165. (1/n expansion of wide networks; cumulant hierarchy.)
- Edgeworth / Gram–Charlier expansions: e.g. P. McCullagh (1987), *Tensor Methods in Statistics*,
  Chapman & Hall (cumulant tensors, Edgeworth series).
- Strassen (1969), Winograd variant: V. Strassen, Gaussian elimination is not optimal,
  *Numerische Mathematik* 13, 354–356.
