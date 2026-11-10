"""Print the '+' lines model.patch adds to a slot's kept-additive test files (offline)."""
import sys, json, re
sys.dont_write_bytecode = True
sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/tools")
sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth")
import grading_repair as gr
plan = gr.load_plan()
run, task, slot = sys.argv[1:4]
only = sys.argv[4] if len(sys.argv) > 4 else None
entry = gr.plan_entries(plan, run, [task], False)[0]
rec = {r["slot"]: r for r in gr.probe_records(entry["probe"])[task]}[slot]
patch = rec["patch"]
add = gr.NEW.additive_paths(patch)
for block in re.split(r"(?m)^(?=diff --git )", patch):
    names = gr.NEW.touched_paths(block)
    if len(names) == 1 and names[0] in add and gr.matching(names, entry["globs"]) and names[0] not in entry["test_patch_paths"]:
        if only and only not in names[0]:
            continue
        plus = [l for l in block.split("\n") if l.startswith("+") and not l.startswith("+++")]
        print(f"=== {names[0]}: +{len(plus)} lines")
        for l in plus[:int(sys.argv[5]) if len(sys.argv) > 5 else 12]:
            print("   ", l[:150])
