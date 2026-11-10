import json, glob, sys, os, csv
from multiprocessing import Pool
R='/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/runs/'
RUNS=['sft-super-kimi-v2-emptythink-wsd-5ep-iter38-rep5-out131k-lenient-think','sft-super-kimi-v2-emptythink-wsd-10ep-iter76-rep5-out131k-lenient-think']
T=['scriggo-method-declarations','wasmi-trap-coredumps','boa-hierarchical-evaluation-cancellation']
def rows(run):
    out=[]
    for f in sorted(glob.glob(R+run+'/shards/shard_*/rollouts.jsonl')):
        for line in open(f):
            # cheap prefilter
            if not any(t in line for t in T): continue
            r=json.loads(line)
            if r.get('task_id') in T:
                out.append(dict(run=run,task=r['task_id'],ri=r['_ng_rollout_index'],reward=r['reward'],pb=r.get('model_patch_bytes'),fp=r.get('opencode_results_fpath'),f2p=(r.get('f2p_passed'),r.get('f2p_total')),fail=r.get('failure_reason'),verr=r.get('verifier_error')))
    return out
def measure(d):
    fp=d['fp']
    if not fp or not os.path.exists(fp): d.update(peak=None); return d
    e=json.load(open(fp)); sid=e['info']['id']
    peak=0;calls=0;comp=0
    for m in e['messages']:
        i=m['info']
        if i.get('role')!='assistant' or i.get('sessionID')!=sid: continue
        t=i.get('tokens',{}); c=t.get('cache',{})
        peak=max(peak,t.get('input',0)+c.get('read',0)+c.get('write',0)+t.get('output',0))
        if i.get('mode')=='compaction' or i.get('summary') is True: comp+=1
        else: calls+=1
    d.update(peak=peak,calls=calls,comp=comp); return d
if __name__=='__main__':
    allr=[]
    with Pool(2) as p:
        for o in p.map(rows,RUNS): allr+=o
    with Pool(8) as p: res=p.map(measure,allr)
    res.sort(key=lambda d:(d['run'],d['task'],d['ri']))
    for d in res:
        print(d['run'][27:31],d['task'][:8],'r%d'%d['ri'],'rew',d['reward'],'pb',d['pb'],'f2p',d['f2p'],'peak',d['peak'],'calls',d.get('calls'),'comp',d.get('comp'),'fail',str(d['fail'])[:60],'verr',str(d['verr'])[:60])
    json.dump(res,open(os.path.dirname(os.path.abspath(__file__))+'/verify_evalruns.json','w'),indent=1,default=str)
