import json,sys,statistics
D='/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/data/targets/sc14-out131k-kimi/records/'
tasks=['scriggo-method-declarations','wasmi-trap-coredumps','boa-hierarchical-evaluation-cancellation']
pick={('scriggo-method-declarations','sc14-r0'),('boa-hierarchical-evaluation-cancellation','sc14-r4'),('wasmi-trap-coredumps','sc14-r1')}
for t in tasks:
    cnt={}
    for line in open(D+t+'.jsonl'):
        r=json.loads(line)
        m=r['model']; cnt.setdefault(m,[0,0,[]]); cnt[m][1]+=1
        if r['status']=='pass': cnt[m][0]+=1
        cnt[m][2].append((r['slot'],r['status'],r['reward']))
        if (t,r['slot']) in pick:
            e=json.load(open(r['source']['opencode_results_fpath']))
            msgs=e['messages']
            sids=set(mm['info'].get('sessionID') for mm in msgs)
            peak=0;nc=0;ncomp=0
            for mm in msgs:
                i=mm['info']
                if i.get('role')!='assistant': continue
                tk=i.get('tokens') or {}
                c=tk.get('input',0)+tk.get('output',0)+(tk.get('cache') or {}).get('read',0)+(tk.get('cache') or {}).get('write',0)
                peak=max(peak,c)
                if i.get('mode')=='compaction' or i.get('summary') is True: ncomp+=1
                else: nc+=1
            print('EXPORT',t,r['slot'],'sessions',len(sids),'peak',peak,'calls',nc,'comp',ncomp,'patchB',len(r['patch'].encode()),r['agent_error'],r['f2p'])
    for m,(p,n,l) in cnt.items(): print('RES',t,m,f'{p}/{n}',l)
