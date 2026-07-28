import json, yaml, glob
BASE='/home/User17/bias_benchmark/nazar_folder/Act2Answer/outputs'
N=55
def sy(path):
    f=glob.glob(f'{BASE}/{path}/**/stats.yaml',recursive=True)
    return yaml.safe_load(open(f[0])) if f else None

def vla_metrics(d, soft=False):
    li=d['last_info']; L=R=Z=succ=0
    ak='chosen_side_soft' if soft else 'chosen_side'
    sk='success_soft_answer' if soft else 'success'
    for i in range(N):
        info=li.get(i) or li.get(str(i))
        if not info: continue
        cs=info.get(ak, info['chosen_side']); s=info.get(sk, info['success'])
        succ+=bool(s)
        if cs==1: L+=1
        elif cs==2: R+=1
        else: Z+=1
    ans=L+R
    return dict(ans_pct=100*ans/N, succ_pct=100*succ/N, right_pct=(100*R/ans if ans else 0), left=L,right=R,none=Z)

def vlm_metrics(jpath):
    d=json.load(open(jpath))
    L=R=0
    for r in d:  # обе раскладки вместе
        ps=r.get('parsed_side')
        if ps=='Left': L+=1
        elif ps=='Right': R+=1
    ans=L+R
    return dict(ans_pct=100*ans/len(d), right_pct=(100*R/ans if ans else 0), left=L, right=R)

print('=== POSITIONAL BIAS: right% среди ОТВЕТИВШИХ (обе раскладки) ===')
print(f"{'режим':16}{'датасет':10}{'ans%':>7}{'right%':>8}  (L/R)")
# VLA hard + soft
for model,pair in (('magma',('bias-magma-pairs_bias','crop-magma-pairs_bias_crop','softcrop-magma')),
                   ('spatialvla',('bias-sv-pairs_bias','crop-spatialvla-pairs_bias_crop','softcrop-spatialvla'))):
    orig,crop,soft=pair
    for tag,run,soft_f in ((f'{model} VLA-hard','orig',(orig,False)),
                            (f'{model} VLA-hard','crop',(crop,False)),
                            (f'{model} VLA-soft','crop',(soft,True))):
        pass
# явно перечислим
rows=[]
def add(mode,ds,path,soft):
    Ln=Rn=succ=ans=0
    agg={'ans':0,'right':0,'left':0,'n':0,'succ':0}
    for sw in ('noswap','swap'):
        p = path if 'bias-' in path or 'crop-' in path else f'{path}-{sw}'
        # hard orig/crop runs имеют оба свои имена уже; soft тоже
        # унифицируем: для VLA путь-до-раны с раскладкой
    return
# проще: перечислить пути напрямую
VLA={
 ('magma','VLA-hard','orig'):['bias-magma-pairs_bias-noswap','bias-magma-pairs_bias-swap'],
 ('magma','VLA-hard','crop'):['crop-magma-pairs_bias_crop-noswap','crop-magma-pairs_bias_crop-swap'],
 ('magma','VLA-soft','crop'):['softcrop-magma-noswap','softcrop-magma-swap'],
 ('spatialvla','VLA-hard','orig'):['bias-sv-pairs_bias-noswap','bias-sv-pairs_bias-swap'],
 ('spatialvla','VLA-hard','crop'):['crop-spatialvla-pairs_bias_crop-noswap','crop-spatialvla-pairs_bias_crop-swap'],
 ('spatialvla','VLA-soft','crop'):['softcrop-spatialvla-pairs_bias_crop-noswap','softcrop-spatialvla-pairs_bias_crop-swap'],
}
for (m,mode,ds),paths in VLA.items():
    soft = (mode=='VLA-soft')
    ans=right=left=succ=n=0
    for p in paths:
        d=sy(p)
        if not d: continue
        r=vla_metrics(d,soft)
        left+=r['left']; right+=r['right']; ans+=(r['left']+r['right']); n+=N; succ+=r['succ_pct']*N/100
    if ans:
        print(f"{m+' '+mode:16}{ds:10}{100*ans/n:7.1f}{100*right/ans:8.1f}  ({left}/{right})")
# VLM
for m in ('magma','paligemma'):
    for ds,assets in (('orig','pairs_bias'),('crop','pairs_bias_crop')):
        r=vlm_metrics(f'{BASE}/vlm-{m}-{assets}.json')
        print(f"{m+' VLM':16}{ds:10}{r['ans_pct']:7.1f}{r['right_pct']:8.1f}  ({r['left']}/{r['right']})")
