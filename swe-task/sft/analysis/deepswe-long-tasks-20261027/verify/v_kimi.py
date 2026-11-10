import json
D='/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/data/targets/sc14-out131k-kimi/records/'
want={}
for t in ['scriggo-method-declarations','wasmi-trap-coredumps','boa-hierarchical-evaluation-cancellation']:
    for line in open(D+t+'.jsonl'):
        r=json.loads(line)
        if r['slot'] in ('kimi-r0',): want[r['source']['line']]=(t,r['slot'],sum(1 for o in r['output'] if o.get('type')=='function_call'))
calls={k:[] for k in want}
with open('/lustre/fsw/portfolios/nemotron/users/hengyuf/deepswe-partial-analysis/verify-raw/pass1/calls.jsonl') as f:
    for line in f:
        li=int(line[7:line.index(',')])
        if li in calls:
            c=json.loads(line); calls[li].append(c)
for li,cs in calls.items():
    cs.sort(key=lambda c:c['ci'])
    allpeak=max(c['ti']+c['to'] for c in cs)
    # main session: chain starting at first n_msgs==2 call, next n_msgs = n + 1 + ntc
    main=[cs[0]]; exp=cs[0]['req']['n_msgs']+1+(cs[0]['ntc'] or 0)
    for c in cs[1:]:
        if c['req']['n_msgs']==exp: main.append(c); exp=c['req']['n_msgs']+1+(c['ntc'] or 0)
    print(want[li],'li',li,'allcalls',len(cs),'all_peak',allpeak,'main_calls',len(main),'main_peak',max(c['ti']+c['to'] for c in main),'main_tc',sum(c['ntc'] or 0 for c in main),'n_msgs2_starts',sum(c['req']['n_msgs']==2 for c in cs))
