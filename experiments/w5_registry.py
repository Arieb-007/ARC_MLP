"""Write results/<date>_w5_term_registry.{md,json}: every ReLU-step diagram term of the dense Kt=4,
cap=3 reference with its metadata (order in 1/n, coherent/random-sign, inputs, W5 group, cost)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arcmlp.reference import dense_cumulants as dc
from arcmlp.compress.cost_model import term_families, family_cost, ENG
from arcmlp.eval.local_eval import provenance
from w5_selection import group_of
day = time.strftime("%Y-%m-%d", time.gmtime())
reg = dc.build_registry(4, 3)
rows = []
for t in reg:
    fams = sorted(term_families(t))
    rows.append(dict(tid=t.tid, output_slice=str(t.p), output_order=sum(t.p), inputs=list(t.inputs),
                     diagram=[list(v) for v in t.diagram], count=t.count, coef=dc.vec_part_coef(t.diagram),
                     cost=t.cost, order_in_1_over_n=t.order_in_1_over_n, coherent=t.coherent,
                     group=group_of(t) or "backbone", families=fams,
                     standalone_billed_units_n1024=round(ENG * sum(family_cost(f) for f in fams), 1)))
json.dump(dict(rows=rows, provenance=provenance(".")), open(f"results/{day}_w5_term_registry.json", "w"), indent=1)
L = [f"# W5 term registry — dense reference Kt = 4, cap = 3 ({day})", "",
     "> LLM-generated (Claude Code) from `arcmlp/reference/dense_cumulants.py` (`build_registry(4, 3)`) and",
     "> `arcmlp/compress/cost_model.py`; generator `experiments/w5_registry.py`.", "",
     f"{len(rows)} terms. A term = (output power-cumulant slice p over len(p) distinct neurons, isomorphism class of a",
     "connected 2-mixed block multiset). 'order' = Σ cost/2: the term is O(n^-order) per entry (counting rule:",
     "|κ_v| = O(n^{-(|v|-1-[v even])/2}); a covariance edge is n^{-1/2}). 'coherent' = every random-sign block",
     "appears an even number of times on the same support, other blocks paired (e.g. c_ij², κ4(2,2)).",
     "'units' = billed cost at n = 1024, L = 16 of the families this term needs alone (cost model, a hypothesis;",
     "families are shared, so costs of a selection are not additive).", "",
     "| # | output | diagram (blocks over output neurons) | count | coef | inputs | order 1/n | coherent | group | families | units |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(rows):
    L.append(f"| {i} | {r['output_slice']} | {r['diagram']} | {r['count']} | {r['coef']:.4g} | {', '.join(r['inputs']) or 'Gaussian'} | "
             f"{r['order_in_1_over_n']:.1f} | {'yes' if r['coherent'] else 'no'} | {r['group']} | {', '.join(r['families'])} | {r['standalone_billed_units_n1024']} |")
open(f"results/{day}_w5_term_registry.md", "w").write("\n".join(L) + "\n")
print(len(rows), "terms written")
