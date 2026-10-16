"""Write results/YYYY-MM-DD_<name>.json and a short .md next to it."""
from __future__ import annotations

import json
import os
import time


def write_result(name: str, payload: dict, md_extra: str = "", results_dir: str = "results") -> str:
    os.makedirs(results_dir, exist_ok=True)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    base = os.path.join(results_dir, f"{day}_{name}")
    with open(base + ".json", "w") as f:
        json.dump(payload, f, indent=1, default=str)
    prov = payload.get("provenance", {})
    lines = [f"# {name}", "", f"- date (UTC): {prov.get('utc', day)}",
             f"- commit: `{prov.get('commit')}`{' (dirty)' if prov.get('dirty') else ''}",
             f"- versions: {prov.get('versions')}", f"- host: {prov.get('host')} ({prov.get('cpu_count')} cpus)", ""]
    s = payload.get("summary")
    if s and "rows" in s:
        lines += ["| idx | name | final MSE | C/B | adjusted | wall s | residual s | failed |", "|---|---|---|---|---|---|---|---|"]
        for r in s["rows"]:
            lines.append(f"| {r['idx']} | {r['name']} | {r['mse']:.4e} | {r['cb']:.4f} | {r['adjusted']:.4e} | "
                         f"{r['wall']:.1f} | {r['residual']:.3f} | {r['failed']} |")
        lines += ["", f"aggregate: `{json.dumps(s['agg'])}`"]
    if md_extra:
        lines += ["", md_extra]
    with open(base + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    return base
