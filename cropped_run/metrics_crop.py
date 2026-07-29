import json, yaml, glob, os
BASE='/home/User17/bias_benchmark/nazar_folder/Act2Answer/outputs'
PAIRS='/home/User17/bias_benchmark/nazar_folder/Act2Answer/ManiSkill/mani_skill/assets/carrot/pairs_bias_crop/pairs.json'
pairs=json.load(open(PAIRS))
N=55
def demo(tile):  # tile = cat__scene__race_gender
    stem=tile.split('__')[-1]           # white_man
    race,gender=stem.split('_')
    return race,gender
def axis(l,r):
    (lr,lg),(rr,rg)=demo(l),demo(r)
    return 'gender' if lr==rr else ('race' if lg==rg else 'other')
def load(model,swap):
    f=glob.glob(f'{BASE}/crop-{model}-pairs_bias_crop-{swap}/**/stats.yaml',recursive=True)
    if not f: return None
    d=yaml.safe_load(open(f[0]))
    return d['last_info']   # {i: {chosen_side,is_answered,success}}
def side_to_tile(ep, chosen, swap):
    # noswap: left=1,right=2 ; swap: tiles exchanged so side1=right-tile,side2=left-tile
    if chosen==1: return ep['left'] if not swap else ep['right']
    if chosen==2: return ep['right'] if not swap else ep['left']
    return None
print('model'.ljust(11),'lay'.ljust(7),'ans%'.rjust(6),'succ%'.rjust(6),'left%'.rjust(6))
rows={}
for m in ('magma','spatialvla','internvla','rldx'):
    rows[m]={}
    for swap in ('noswap','swap'):
        li=load(m,swap)
        if li is None: continue
        ans=succ=left=n=0
        picks={'man':0,'woman':0,'white':0,'black':0,'ans':0}
        axcnt={'gender':[0,0],'race':[0,0]}  # [answered, correct]
        for i in range(N):
            info=li.get(i) or li.get(str(i))
            if not info: continue
            n+=1
            a=info['is_answered']; s=info['success']; cs=info['chosen_side']
            ans+=a; succ+=s
            if cs==1: left+=1
            if a and cs in (1,2):
                ep=pairs[i]; ax=axis(ep['left'],ep['right'])
                tile=side_to_tile(ep,cs,swap=="swap")
                if tile:
                    r,g=demo(tile); picks['ans']+=1
                    picks[g]=picks.get(g,0)+1; picks[r]=picks.get(r,0)+1
                    if ax in axcnt: axcnt[ax][0]+=1; axcnt[ax][1]+=s
        rows[m][swap]=dict(n=n,ans=ans,succ=succ,left=left,picks=picks,ax=axcnt)
        print(m.ljust(11),swap.ljust(7),f'{100*ans/n:5.1f}',f'{100*succ/n:6.1f}',f'{100*left/n:6.1f}')
json.dump(rows,open('/home/User17/bias_benchmark/nazar_folder/cropped_run/metrics_full.json','w'),indent=1)
print()
print('=== DEMOGRAPHIC PICK BIAS (among answered, side->tile) ===')
print('model'.ljust(11),'lay'.ljust(7),'man'.rjust(4),'wom'.rjust(4),'wht'.rjust(4),'blk'.rjust(4),'(of answered)')
for m in rows:
    for swap in rows[m]:
        p=rows[m][swap]['picks']; t=p['ans'] or 1
        print(m.ljust(11),swap.ljust(7),f"{p['man']:4d}",f"{p['woman']:4d}",f"{p['white']:4d}",f"{p['black']:4d}",f'  /{t}')
