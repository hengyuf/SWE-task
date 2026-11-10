import json, glob, os, sys, csv, statistics
from multiprocessing import Pool
RUNS='/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/runs'
A='/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/analysis/deepswe-long-tasks-20261027'
T=['scriggo-method-declarations','wasmi-trap-coredumps','boa-hierarchical-evaluation-cancellation']
def scan(path):
    out=[]
    with open(path,'rb') as f:
        for line in f:
            if not any(t.encode() in line for t in T): continue
            d=json.loads(line)
            tid=d.get('task_id') or d.get('responses_create_params',{}).get('metadata',{}).get('task_id')
            if tid not in T: continue
            out.append(dict(path=path,task_id=tid,ti=d.get('_ng_task_index'),ri=d.get('_ng_rollout_index'),reward=d.get('reward'),patch=d.get('model_patch_bytes'),fp=d.get('opencode_results_fpath')))
    return out
def length(fp):
    d=json.load(open(fp)); sid=d['info']['id']; pk=0; nc=0; ncomp=0
    for m in d['messages']:
        i=m['info']
        if i.get('sessionID')!=sid or i.get('role')!='assistant': continue
        t=i.get('tokens',{}); c=t.get('cache',{})
        pk=max(pk,t.get('input',0)+c.get('read',0)+c.get('write',0)+t.get('output',0))
        if i.get('mode')=='compaction' or i.get('summary') is True: ncomp+=1
        else: nc+=1
    return pk,nc,ncomp
if __name__=='__main__':
    runs=sys.argv[1:]
    files=[(r,p) for r in runs for p in sorted(glob.glob(f'{RUNS}/{r}/shards/shard_*/rollouts.jsonl'))]
    with Pool(8) as P: res=P.map(scan,[p for _,p in files])
    rep={}
    for (r,_),rows in zip(files,res):
        for x in rows: x['run']=r; rep[(r,x['task_id'],x['ri'])]=x  # last row wins (file order)
    print(json.dumps({'nrows':{r:sum(1 for k in rep if k[0]==r) for r in runs}}))
    for r in runs:
        pt={row['task_id']:row for row in csv.DictReader(open(f'{RUNS}/{r}/per_task.csv'))}
        sv={}
        sp=f'{A}/{r}.jsonl'
        if os.path.exists(sp):
            for l in open(sp):
                d=json.loads(l)
                if d['task_id'] in T: sv[(d['task_id'],d['rollout_index'])]=d
        for t in T:
            xs=[v for k,v in rep.items() if k[0]==r and k[1]==t]
            res_=sum(1 for v in xs if (v['reward'] or 0)>=1)
            svres=sum(1 for v in sv.values() if v['task_id']==t and v.get('resolved'))
            print(r,t,'raw',f'{res_}/{len(xs)}','per_task',pt[t]['resolved']+'/'+pt[t]['completed'],'saved',svres, 'rewards',sorted((v['ri'],v['reward']) for v in xs))
    # lengths
    picks=[tuple(a.split(':')) for a in os.environ.get('PICKS','').split(',') if a]
    for r,t,ri in picks:
        x=rep[(r,t,int(ri))]
        fp=x['fp']; ok=fp and os.path.exists(fp)
        L=length(fp) if ok else None
        s=None
        sp=f'{A}/{r}.jsonl'
        for l in open(sp):
            d=json.loads(l)
            if d['task_id']==t and d['rollout_index']==int(ri): s=(d.get('peak_ctx'),d.get('n_calls'),d.get('n_compactions'),d.get('export_path'))
        print('LEN',r,t,ri,'mine',L,'saved',s,'fp_match',s and s[3]==fp)
