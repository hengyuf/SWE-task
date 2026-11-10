import csv,json,os,sys
R="/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/runs"
T=["scriggo-method-declarations","wasmi-trap-coredumps","boa-hierarchical-evaluation-cancellation"]
out=[]
for n in sorted(os.listdir(R)):
    p=os.path.join(R,n)
    if not os.path.isdir(p) or not n.startswith(("sft-","base-","r2-","r3-","r4-","r5-")): continue
    d={"run":n}
    try: m=json.load(open(p+"/metrics.json"))
    except Exception as e: m={}; d["metrics_err"]=str(e)
    c=m.get("coverage",{})
    d["resolved_at_1"]=m.get("resolved@1"); d["complete"]=c.get("rollouts_left_to_collect")==0 and not c.get("partial",True)
    d["coverage"]=c
    try: d["model_path"]=json.load(open(p+"/run_config.json")).get("model_path")
    except Exception as e: d["model_path"]=m.get("model_path")
    d["tasks"]=[]
    if os.path.exists(p+"/per_task.csv"):
        for r in csv.DictReader(open(p+"/per_task.csv")):
            if r["task_id"] in T: d["tasks"].append(r)
    out.append(d)
tg=list(csv.DictReader(open("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/data/targets/sc14-out131k-kimi/per_task.csv")))
json.dump({"runs":out,"targets_cols":list(tg[0].keys()),"targets":[r for r in tg if r.get("task_id") in T]},open(os.path.dirname(os.path.abspath(__file__))+"/acc_table.json","w"),indent=1)
def s(k): return sum(int(r[k] or 0) for r in tg)
print(len(tg),{k:s(k) for k in tg[0] if k.startswith(("sc14","kimi"))})
