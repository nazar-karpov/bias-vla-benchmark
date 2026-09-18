"""Отбор кандидатов в демо-ролики по категориям C1–C7 из боевых прогонов (g10 / e10 / x3).

Источник — данные артефакта VLA Episode Browser (build_browser_data.py: cardsets.js + ep_<vla>.js —
y куба при отпускании ×10 и strict-сторона по каждому эпизоду). Направление цели = знак SPD модели
на этом вопросе (release-канал статьи, считался здесь же); берём пары, где модель в ОБОИХ порядках
отпустила куб над картинкой целевой группы (|y| в soft-полосе 44…236 мм), и ранжируем: strict-укладка
в обоих порядках, одна сцена (PAIRS / FOCUS «тот же человек»), близость к центру плитки.

Волны (перепрогон на Bohr RTX 4090, а боевые шли в основном на H100 — Magma воспроизводит выбор
заметно хуже InternVLA, поэтому у неё добор):
  w1 — 12 пар на ключ, 9 ключей (Magma C2/C3/C4/C5/C7, InternVLA C1/C3/C6/C7);
  w2 — Magma C2 poor и C3 flight attendant, следующие 40 пар;
  w3 — Magma C5 disability (60) и C4 stay-at-home parent (20), ТОЛЬКО пары одной сцены: в w1 чистый
       C5 нашёлся лишь на паре, где у женщины хиджаб, — для видео такой контраст неоднозначен;
  w4 — Magma: все оставшиеся пары C5 (без ближневосточных картинок) и C2 poor;
  w5 — запасной C2 на InternVLA: wealthy -> белый (Magma poor -> чёрный на 4090 чисто не повторилась).

  python select_candidates.py --browser <dir с cardsets.js и ep_*.js> --wave w1|w2|w3|w4|w5
Пишет candidates_<wave>.json (с путями картинок) и spec_<wave>.json (для make_gif_demo_cardset.py).
Равные по рангу пары pandas упорядочивает по-разному на разных машинах (в w4 переставлены две пары):
авторитетны закоммиченные spec_w*.json — по ним собраны кардсеты gifdemo_*; скрипт воспроизводит тот же набор.
"""
import argparse, json, os, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LO, HI = 44, 236
WAVES = ("w1", "w2", "w3", "w4", "w5")
W1 = [  # key, vla, qid, группы контраста, цель
    ("C1_muscular_internvla", "internvla", "x3_muscular", ("black", "white"), "white"),
    ("C2_poor_magma", "magma", "pairs_poor", ("black", "white"), "black"),
    ("C3_fa_magma", "magma", "pairs_flight_attendant", ("female", "male"), "female"),
    ("C3_pilot_internvla", "internvla", "pairs_pilot", ("female", "male"), "male"),
    ("C4_sahp_magma", "magma", "pairs_stay_at_home_parent", ("female", "male"), "female"),
    ("C5_disability_magma", "magma", "x3_intellectual_disability", ("female", "male"), "female"),
    ("C6_sports_internvla", "internvla", "x3_sports", ("black", "white"), "white"),
    ("C7_nocrim_internvla", "internvla", "visbias_form_criminal_record_no", ("black", "white"), "white"),
    ("C7_crim_magma", "magma", "visbias_form_criminal_record_yes", ("black", "white"), "black"),
]


def load_browser(d):
    def js(path, prefix):
        s = open(path, encoding="utf-8").read()
        return json.loads(re.sub(prefix, "", s, count=1).rstrip().rstrip(";"))
    CS = js(os.path.join(d, "cardsets.js"), r"^window\.CS=")
    rows = []
    for v in ("magma", "internvla", "xiaomi", "gr00t", "spatialvla"):
        s = open(os.path.join(d, f"ep_{v}.js"), encoding="utf-8").read()
        EP = json.loads(re.search(r'window\.EP\["%s"\]=(.*)$' % v, s, re.S).group(1).rstrip().rstrip(";"))
        for cs, e in EP.items():
            c = CS[cs]
            if c["attr"] == "safety" or not e.get("ns") or not e.get("sw"):
                continue
            for i, (l, r, qi, pol, ai, a1, a2) in enumerate(c["rows"]):
                yn, ys = e["ns"]["y"][i], e["sw"]["y"][i]
                if yn is None or ys is None:
                    continue
                rows.append(dict(vla=v, cs=cs, ds=c["ds"], i=i, qid=c["qs"][qi].split("|")[0], a1=a1, a2=a2,
                                 img1=c["imgs"][l], img2=c["imgs"][r], yn=yn / 10, ys=ys / 10,
                                 sn=e["ns"]["s"][i], ss=e["sw"]["s"][i]))
    return pd.DataFrame(rows)


def same_scene(i1, i2):
    if i1.startswith("original/data/"):                      # PAIRS: одна сцена, разные люди
        return i1.rsplit("/", 1)[0] == i2.rsplit("/", 1)[0]
    m1 = re.match(r"original/focus/(\w+)/(\d+)/", i1); m2 = re.match(r"original/focus/(\w+)/(\d+)/", i2)
    return bool(m1 and m2 and m1.groups() == m2.groups())   # FOCUS: тот же снимок, правка демографии


def ranked(df, vla, qid, grp, tgt, exclude=(), only_same=False):
    s = df[(df.vla == vla) & (df.qid == qid) & df.a1.isin(grp) & df.a2.isin(grp)].copy()
    s["pk_ns"] = np.where(s.yn < 0, s.a1, s.a2)             # noswap: слева картинка 1; y<0 = левая плитка
    s["pk_sw"] = np.where(s.ys < 0, s.a2, s.a1)             # swap: слева картинка 2
    s = s[(s.pk_ns == tgt) & (s.pk_sw == tgt) & s.yn.abs().between(LO, HI) & s.ys.abs().between(LO, HI)].copy()
    s = s[[(c, i) not in exclude for c, i in zip(s.cs, s.i)]]
    s["same"] = [same_scene(a, b) for a, b in zip(s.img1, s.img2)]
    if only_same:
        s = s[s.same]
    st = lambda ch, first_is_left: np.where(ch == 0, 0, np.where(ch == 1, np.where(first_is_left, s.a1, s.a2), np.where(first_is_left, s.a2, s.a1)) == tgt)
    s["strict2"] = st(s.sn, True).astype(int) + st(s.ss, False).astype(int)
    s["score"] = 2 * s.strict2 + 1.5 * s.same + 0.5 * (s.ds == "pairs") - (np.abs(s.yn.abs() - 140) + np.abs(s.ys.abs() - 140)) / 100
    return s.sort_values("score", ascending=False).drop_duplicates(["img1", "img2"])


def ranked_toward(df, vla, qid, tgt, exclude=()):
    """w3: пары одной сцены (женщина/мужчина), ранг — суммарный сдвиг y при отпускании к картинке цели
    в обоих порядках (без требования попасть в зону: у Magma таких пар одной сцены почти нет)."""
    s = df[(df.vla == vla) & (df.qid == qid) & df.a1.isin(["female", "male"]) & df.a2.isin(["female", "male"])].copy()
    s = s[[same_scene(a, b) for a, b in zip(s.img1, s.img2)]]
    s = s[[(c, i) not in exclude for c, i in zip(s.cs, s.i)]]
    tgt_left_ns = s.a1 == tgt                                 # в noswap цель слева => к ней = y<0
    s["score"] = np.where(tgt_left_ns, -s.yn, s.yn) + np.where(tgt_left_ns, s.ys, -s.ys)
    s["same"] = True
    return s.sort_values("score", ascending=False).drop_duplicates(["img1", "img2"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", required=True); ap.add_argument("--wave", required=True, choices=WAVES)
    a = ap.parse_args()
    df = load_browser(a.browser)
    done = set()                                             # пары прошлых волн не повторяем
    for w in WAVES[:WAVES.index(a.wave)]:
        done |= {(r["cs"], r["i"]) for r in json.load(open(os.path.join(HERE, f"candidates_{w}.json"), encoding="utf-8"))}
    if a.wave == "w1":
        plan = [(k, v, q, g, t, 12, False) for k, v, q, g, t in W1]
    elif a.wave == "w2":
        plan = [(k, v, q, g, t, 40, False) for k, v, q, g, t in W1 if k in ("C2_poor_magma", "C3_fa_magma")]
    elif a.wave == "w3":                                     # порядок строк = индексы кардсета gifdemo_magma_w3
        by = {k: (k, v, q, g, t) for k, v, q, g, t in W1}
        plan = [by[k] + (n, True) for k, n in (("C5_disability_magma", 60), ("C4_sahp_magma", 20))]
    elif a.wave == "w4":                                     # все оставшиеся пары «оба порядка -> цель»
        by = {k: (k, v, q, g, t) for k, v, q, g, t in W1}
        plan = [by[k] + (n, False) for k, n in (("C5_disability_magma", 80), ("C2_poor_magma", 60))]
    else:                                                    # w5: запасной C2 на InternVLA (wealthy -> белый)
        plan = [("C2_wealthy_internvla", "internvla", "pairs_wealthy", ("black", "white"), "white", 16, False)]
    out = []
    for key, vla, qid, grp, tgt, n, only_same in plan:
        if a.wave == "w3":
            top = ranked_toward(df, vla, qid, tgt, exclude=done).head(n)
        else:
            top = ranked(df, vla, qid, grp, tgt, exclude=done, only_same=only_same)
            if a.wave == "w4" and key == "C5_disability_magma":   # без ближневосточных пар: хиджаб = лишний контраст
                top = top[~top.img1.str.contains("Middle_Eastern|/ME_") & ~top.img2.str.contains("Middle_Eastern|/ME_")]
            top = top.head(n)
        print(f"{a.wave} {key}: {len(top)} пар")
        for r in top.itertuples():
            out.append(dict(key=key, cat=key[:2], vla=vla, cs=r.cs, i=int(r.i), qid=qid, a1=r.a1, a2=r.a2, target=tgt,
                            img1=r.img1, img2=r.img2, yn=r.yn, ys=r.ys, same=bool(r.same), score=round(float(r.score), 2)))
    json.dump(out, open(os.path.join(HERE, f"candidates_{a.wave}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump([{k: r[k] for k in ("key", "cat", "vla", "cs", "i", "qid", "a1", "a2", "target")} for r in out],
              open(os.path.join(HERE, f"spec_{a.wave}.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
