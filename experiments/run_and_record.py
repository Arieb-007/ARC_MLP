"""Run an experiment script verbatim, capture stdout, and write results/YYYY-MM-DD_<name>.{md,json}
with provenance. Usage: python experiments/run_and_record.py experiments/predictability.py e1_predictability"""
import json, subprocess, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcmlp.eval.local_eval import provenance
script, name = sys.argv[1], sys.argv[2]
t0 = time.time()
p = subprocess.run([sys.executable, script], capture_output=True, text=True)
dt = time.time() - t0
day = time.strftime("%Y-%m-%d", time.gmtime())
prov = provenance(".")
base = f"results/{day}_{name}"
json.dump(dict(script=script, returncode=p.returncode, seconds=dt, stdout=p.stdout, stderr=p.stderr[-3000:],
               provenance=prov), open(base + ".json", "w"), indent=1)
open(base + ".md", "w").write(f"# {name}\n\n- script: `{script}` (verbatim from the brief; seeds hard-coded in the script)\n"
    f"- commit: `{prov['commit']}`{' (dirty)' if prov['dirty'] else ''}; versions: {prov['versions']}\n"
    f"- host: {prov['host']} ({prov['cpu_count']} cpus), {prov['utc']}, runtime {dt:.0f} s, exit {p.returncode}\n\n"
    f"## stdout\n\n```\n{p.stdout}```\n" + (f"\n## stderr (tail)\n\n```\n{p.stderr[-1500:]}\n```\n" if p.returncode else ""))
print(f"{name}: exit {p.returncode}, {dt:.0f}s -> {base}.md")
