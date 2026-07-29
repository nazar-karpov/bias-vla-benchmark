import yaml, glob
BASE='/home/User17/bias_benchmark/nazar_folder/Act2Answer/outputs'
N=55
def sy(path):
    f=glob.glob(f'{BASE}/{path}/**/stats.yaml',recursive=True)
    return yaml.safe_load(open(f[0])) if f else None
def metrics(d):
    li=d['last_info']; st=d['stats']
    ans=succ=left=right=0; soft=st.get('is_success_soft_v1',0.0)
    for i in range(N):
        info=li.get(i) or li.get(str(i))
        if not info: continue
        ans+=info['is_answered']; succ+=info['success']
        if info['chosen_side']==1: left+=1
        elif info['chosen_side']==2: right+=1
    anspct=100*ans/N; succpct=100*succ/N
    rl = 100*right/(left+right) if (left+right) else 0
    return anspct, succpct, 100*soft, rl
runs={
 'magma-orig': ('bias-magma-pairs_bias-noswap','bias-magma-pairs_bias-swap'),
 'magma-crop': ('crop-magma-pairs_bias_crop-noswap','crop-magma-pairs_bias_crop-swap'),
 'sv-orig':    ('bias-sv-pairs_bias-noswap','bias-sv-pairs_bias-swap'),
 'sv-crop':    ('crop-spatialvla-pairs_bias_crop-noswap','crop-spatialvla-pairs_bias_crop-swap'),
}
print(f"{'run':14}{'lay':8}{'ans%':>6}{'succ%':>7}{'soft%':>7}{'right%':>8}  (right% среди ответивших)")
for name,(ns,sw) in runs.items():
    for lab,p in (('noswap',ns),('swap',sw)):
        d=sy(p)
        if not d: print(f'{name:14}{lab:8}  NO DATA'); continue
        a,s,so,r=metrics(d)
        print(f'{name:14}{lab:8}{a:6.1f}{s:7.1f}{so:7.1f}{r:8.1f}')
