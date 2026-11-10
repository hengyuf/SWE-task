# Independent re-check: datasets json loader + a re-implementation of SFTTokenizer.tokenize_conversation (identity)
# and SFTDataset.__getitem__ (CP16 padding, 524288 truncation, step weights), using transformers' tokenizer.
import os,sys,json,glob
os.environ['HF_HUB_OFFLINE']='1'; os.environ['HF_HOME']=os.environ['TMPDIR']+'/hf'; os.environ['HF_DATASETS_CACHE']=os.environ['TMPDIR']+'/hf/ds'
import numpy as np, transformers, datasets
from multiprocessing import Pool
H='/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf'
OUT='/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/corpus/kimi-pass-v2-20261023'
IDT="{% for message in messages %}{{ message['content'] }}{% endfor %}"
SEQ=524288; IGN=-100
tok=None
def init():
    global tok; transformers.logging.set_verbosity_error(); tok=transformers.AutoTokenizer.from_pretrained(H)
def tc(conv):
    tokens=np.array(tok.apply_chat_template(conv,tokenize=True,add_generation_prompt=False,return_dict=False,chat_template=IDT))
    if tokens.ndim>1: tokens=tokens[0]
    target=tokens.copy(); idx=0
    for ti,turn in enumerate(conv):
        if turn['role']=='assistant':
            assert len(turn['content'])>0; assert conv[ti-1]['role'] in ('user','tool')
        tt=tok.apply_chat_template([turn],tokenize=True,chat_template=IDT,return_dict=False)
        if isinstance(tt[0],list): tt=tt[0]
        if turn['role'] in ('system','user','tool'): target[idx:idx+len(tt)]=IGN
        assert np.array_equal(tokens[idx:idx+len(tt)],tt)
        idx+=len(tt)
    assert idx==len(tokens)
    return tokens.tolist(),target.tolist()
def row(args):
    path,ri,msgs=args
    pad=tok.convert_tokens_to_ids('<unk>')
    convs=[];cur=[]
    for m in msgs:
        m={k:v for k,v in m.items() if v is not None}
        if m['role']=='system':
            if cur: convs.append(cur)
            cur=[m]
        else: cur.append(m)
    convs.append(cur)
    P=[];W=[];T=[];expect=0.0;ids=[]
    for c in convs:
        t,g=tc(c); s=c[0]
        cnt=[int(x) for x in s['assistant_token_counts']]; w=[float(x) for x in s['assistant_weights']]
        el=[i for i,x in enumerate(g) if x!=IGN]
        assert sum(cnt)==len(el), (s['step_weight_agent_run_id'],sum(cnt),len(el))
        cw=[0.0]*len(g); cur_=0
        for k,ww in zip(cnt,w):
            for p in el[cur_:cur_+k]: cw[p]=ww
            cur_+=k
        expect+=sum(k*ww for k,ww in zip(cnt,w))
        P+=t;T+=g;W+=cw
        r=len(P)%32
        if r: n=32-r; P+=[pad]*n;T+=[pad]*n;W+=[0.0]*n
        assert len(P)<SEQ+1, ('TRUNCATION', path, ri)
        ids.append(s['step_weight_agent_run_id'])
    n=SEQ+1-len(P); P+=[pad]*n;T+=[pad]*n;W+=[0.0]*n
    labels=np.array(T[1:]); lm=np.array(W[1:],dtype=np.float32); lm[labels==pad]=0; lm[labels==IGN]=0
    # first token of a conversation is a system token, so the target shift loses nothing
    return path,ri,len(convs),float(lm.sum()),expect,ids
if __name__=='__main__':
    jobs=[]
    for f in sorted(glob.glob(OUT+'/raw_sft/*.packed.512k')):
        ds=datasets.load_dataset('json',data_files=f,split='all')
        print(os.path.basename(f),'rows',len(ds),flush=True)
        for i in range(len(ds)): jobs.append((os.path.basename(f),i,ds[i]['messages']))
    allids=[];tot=0;exp=0
    with Pool(12,initializer=init) as p:
        for path,ri,nc,s,e,ids in p.imap_unordered(row,jobs):
            assert abs(s-e)<0.5,(path,ri,s,e); tot+=s; exp+=e; allids+=ids
    assert len(allids)==len(set(allids))
    print('rows',len(jobs),'convs',len(allids),'loss_mask_sum',tot,'expected',exp,'OK')
    # v2 additions: every conversation of the shards is in the selection once (and vice versa), with the same token
    # length; v1's 379 conversations are byte-identical to v1 (training_row_sha256 per agent_run_id).
    import hashlib
    sel=[json.loads(l) for l in open(OUT+'/selection/kimi_pass_v2.jsonl')]
    assert sorted(allids)==sorted(s['agent_run_id'] for s in sel)
    shas={}
    for f in sorted(glob.glob(OUT+'/raw_sft/*.packed.512k')):
        for line in open(f,encoding='utf-8'):
            ms=json.loads(line)['messages']; cur=[]
            for m in ms+[{'role':'system','_end':1}]:
                if m['role']=='system' and cur:
                    shas[cur[0]['step_weight_agent_run_id']]=hashlib.sha256(json.dumps({'messages':cur},ensure_ascii=False).encode()).hexdigest(); cur=[]
                cur.append(m)
    assert all(shas[s['agent_run_id']]==s['training_row_sha256'] for s in sel)
    v1=[json.loads(l) for l in open(OUT.replace('-v2-','-v1-')+'/selection/kimi_pass_v1.jsonl')]
    assert all(shas[s['agent_run_id']]==s['training_row_sha256'] for s in v1), 'v1 conversation changed'
    from collections import Counter
    print('selection rows',len(sel),'by source',dict(Counter(s['source'] for s in sel)),'v1 conversations identical',len(v1),'OK')
