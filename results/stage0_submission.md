# Stage 0 — graded Phase-2 submission

> LLM-generated record (Claude Code). Deadline: Phase-2 submissions close **2026-10-17 23:59 UTC**.
> Status: **DONE — four submissions graded** (submitted by Areeb from a Kaggle notebook on
> 2026-10-16; the cloud sandbox has no AIcrowd access).

## Grader results [measured by the grader; transcribed from Areeb's screenshot of the submissions page, 3 significant digits]

| ID | submitted (page time) | status | adjusted | final-layer MSE | bundle (inferred from the score) |
|---|---|---|---|---|---|
| [AIcrowd 336554](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336554) | Oct 16, 17:42 | GRADED | **5.23e-9** | 2.13e-8 | A (V29 + STRASSEN_MIN=8) |
| [AIcrowd 336558](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336558) | Oct 16, 17:51 | GRADED | **5.23e-9** | 2.13e-8 | A (V29 + STRASSEN_MIN=8) |
| [AIcrowd 336561](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336561) | Oct 16, 18:20 | GRADED | 5.42e-9 | 2.13e-8 | B (V29 verbatim) |
| [AIcrowd 336562](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336562) | Oct 16, 18:41 | GRADED | 5.42e-9 | 2.13e-8 | B (V29 verbatim) |

Acceptance criterion of Stage 0 (a graded, non-failed submission ID) is **met**. Best:
**[AIcrowd 336554](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336554) / [AIcrowd 336558](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336558), 5.23e-9 adjusted**.

**Leaf-8 projection vs grader.** Projected ≈ 5.22e-9 (multiplier ≈ 0.245, raw unchanged);
graded 5.23e-9 with raw 2.13e-8 unchanged → implied multiplier 0.2542 × 5.23/5.42 ≈ 0.245
[consistent; the bundle labels are inferred from the scores — confirm via each "View" page /
report.json (`mean_score_multiplier`) before citing an ID in the write-up]. Leaf 8 did not
fail on memory, residual or wall time on the grader (all 4 graded).
Leaderboard top (Areeb, 2026-10-17): 1.42e-9 adjusted — 3.7× below our best.

| field | primary (A) | fallback (B) |
|---|---|---|
| bundle | `submission/estimator.py` | `submission_fallback_v29/estimator.py` |
| content | 504aldo V29 (MIT) + `STRASSEN_MIN` 32 → 8 (one constant) | 504aldo V29 (MIT), code byte-identical; MIT header added |
| packaged tarball sha256 (Python 3.10 packager) | `898df0f9d14bafc6…f6e7` (`submission.tar.gz`) | `c7580aba334e2fb9…a0f6` (`submission_fallback_v29.tar.gz`) |
| files / size | 2 (estimator.py, manifest.json) / 30.0 KB | 2 / 29.9 KB |
| `whest validate` / `validate-package` | pass / pass | pass / pass |
| off-suite smoke shapes 4×2, 64×16, 256×32, 1024×3 | finite, correct shape | (same code path) |
| setup time | 0.10–0.30 s (cap 5 s) | 0.10–0.17 s |
| local C/B (first / steady) | 0.2592 / 0.2432 **[measured]** | 0.2671 / 0.2526 **[measured]**, = grader min/max |
| raw final-layer MSE | ≈ 2.13e-8 **[projection]**: predictions within 1.1e-11 MSE of B | 2.1327e-8 on the 100 public MLPs **[external: grader report [AIcrowd 333467](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/333467)]** |
| adjusted | ≈ 5.22e-9 **[projection]** | 5.418e-9 **[external]** |
| residual s | local 0.505–0.556 (py3.13); grader projection ≈ 0.21–0.23 **[projection]** | grader 0.183 mean / 0.195 max **[external]**; local 0.435–0.48 |
| predict wall s | local 93–101; grader projection ≈ 81 / 89 max | grader 68.9 / 75.4 max |
| peak RSS (local, in-process) | 6.48 GB | 6.00 GB |
| submission ID | [AIcrowd 336554](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336554), [AIcrowd 336558](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336558) (5.23e-9) | [AIcrowd 336561](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336561), [AIcrowd 336562](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336562) (5.42e-9) |
| commit | see `git log -- submission/` | see `git log -- submission_fallback_v29/` |

Evidence: `results/2026-10-16_stage0_leaf_probe.{md,json}` (cost/memory/timing per leaf; prediction
deltas; why local `whest run --runner subprocess` fails even for V29 and is not predictive of
the grader). Python-3.10 check: the leaf-8 file under Python 3.10.20 / numpy 2.2.6 / flopscope
0.12.3 (grader: 3.10.21 / 2.2.6 / 0.12.1) gives identical FLOPs (0.2592 / 0.2442) and final-layer
predictions within 7e-13 MSE of the Python-3.13 run **[measured]**.

Local raw MSE was **not** measured: an MC truth with the noise needed to resolve 2e-8 at
1024×16 (≥ 1e7–1e8 samples per MLP, F6 variance) costs hours per MLP on this 4-core VM and the
public split is not reachable from it. The raw MSE above is the grader's own number for V29
(external archive) and the leaf-8 projection rests on the measured prediction deltas.

## Risks

1. **Memory** (A only): +0.48 GB peak RSS over V29 locally. Grader limit is "8 GB of RAM" for the
   solution process with arrays in a flopscope server; V29 passes there. Projected OK, margin
   reduced. If A fails with a memory error, B is the answer.
2. **Fair accounting**: Strassen–Winograd written as flopscope ops is used by V29 and many graded
   submissions; 504aldo asked ARC for a ruling on 2026-09-05 and had none as of their post. Same
   risk for A and B.
3. **Private rerun**: if final ranking re-runs a nominated submission, B has the larger timing
   margin.

## Exact submit commands (Areeb)

```bash
git clone https://github.com/Arieb-007/ARC_MLP && cd ARC_MLP
git checkout claude/elegant-hypatia-fcf7nb          # or main once merged
python3.10 -m venv .venv && . .venv/bin/activate       # grader runs 3.10; the manifest records the packager's Python
pip install "flopscope>=0.12.1,<0.13" "whestbench>=0.16.0"
whest login                                            # AIcrowd API key: https://www.aicrowd.com/participants/me/edit
cd submission
whest submit --estimator estimator.py --dry-run        # shows files/manifest; expect estimator.py only, python min 3.10
whest submit --estimator estimator.py --description "Arieb-007 S0-A: 504aldo V29 (MIT) + STRASSEN_MIN=8" --watch --watch-timeout 3600
cd ../submission_fallback_v29                          # optional second slot (10/day), or only if A fails
whest submit --estimator estimator.py --description "Arieb-007 S0-B: 504aldo V29 verbatim (MIT)" --watch --watch-timeout 3600
```

Then record the submission IDs, status and grader numbers in this file (or paste them to Claude).

Acceptance (brief §4): a graded, non-failed submission ID here.
