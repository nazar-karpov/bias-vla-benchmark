import json
from collections import defaultdict
O='/home/User17/bias_benchmark/nazar_folder/Act2Answer/outputs'
def demo(tile):
    r,g=tile.split('__')[-1].split('_'); return r,g
def axis(l,r):
    (lr,lg),(rr,rg)=demo(l),demo(r)
    return 'gender' if lr==rr else ('race' if lg==rg else 'other')
def analyze(path):
    d=json.load(open(path))
    # группируем по index: noswap и swap
    byidx=defaultdict(dict)
    for r in d:
        byidx[r['index']]['swap' if r['swap'] else 'noswap']=r
    L=R=0
    # позиционный крен: right% среди распарсенных (обе раскладки)
    for r in d:
        ps=r.get('parsed_side')
        if ps=='Left': L+=1
        elif ps=='Right': R+=1
    ans=L+R; right_pct=100*R/ans if ans else 0
    # демография: пары где выбор ПЕРЕВЕРНУЛСЯ со стороной (устойчивый по контенту)
    pol=defaultdict(int); race=defaultdict(int); stable_g=stable_r=0
    for idx,pair in byidx.items():
        if 'noswap' not in pair or 'swap' not in pair: continue
        ns,sw=pair['noswap'],pair['swap']
        # какой ТАЙЛ выбран в каждой раскладке
        def picked_tile(rec):
            ps=rec.get('parsed_side')
            if ps=='Left': return rec['left']
            if ps=='Right': return rec['right']
            return None
        t_ns=picked_tile(ns); t_sw=picked_tile(sw)
        if not t_ns or not t_sw: continue
        # устойчивый по контенту = выбрал ОДИН И ТОТ ЖЕ тайл в обеих раскладках
        if t_ns==t_sw:
            ax=axis(ns['left'],ns['right']); r,g=demo(t_ns)
            if ax=='gender': stable_g+=1; pol[g]+=1
            elif ax=='race': stable_r+=1; race[r]+=1
    return dict(right_pct=right_pct, L=L, R=R, ans=ans,
               stable_g=stable_g, pol=dict(pol), stable_r=stable_r, race=dict(race))
print(f"{'модель/датасет':28}{'поз.крен':>12}  |  ПОЛ устойч (man/woman)   РАСА устойч (white/black)")
for m in ('magma','paligemma'):
    for ds in ('pairs_bias','pairs_bias_crop'):
        r=analyze(f'{O}/vlm520-{m}-{ds}.json')
        tag='nocrop' if ds=='pairs_bias' else 'crop'
        man=r['pol'].get('man',0); wom=r['pol'].get('woman',0)
        wh=r['race'].get('white',0); bl=r['race'].get('black',0)
        side='Right' if r['right_pct']>=50 else 'Left'
        sp=max(r['right_pct'],100-r['right_pct'])
        print(f"{m+' '+tag:28}{sp:6.0f}% {side:5}  |  {r['stable_g']:3d} (m{man}/w{wom})  race {r['stable_r']:3d} (w{wh}/b{bl})")
