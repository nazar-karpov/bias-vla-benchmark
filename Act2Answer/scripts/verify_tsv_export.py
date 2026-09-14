#!/usr/bin/env python3
"""Независимая проверка TSV-выгрузки export_tsv_vlasim.py (кода экспортёра не импортирует).
1) строки: признаки картинок берутся из ИМЁН ФАЙЛОВ (не из attr_* колонок), answer_side = признак картинки на стороне
   знака y_release_mm; soft/hard согласованы; swap-строка = переставленная noswap; пара в обоих ethnicity-файлах с
   обратным ответом; число строк.
2) агрегаты: Δ pull(pos)−pull(neg) из y_release_mm по темам/контрастам против d в metrics/*.csv (all_metrics, из traj.npz);
   VERI — pull vs 0 по вопросам против *_byq_abs.csv.
3) сырьё: случайные эпизоды из traj.npz → y при разжатии → строка TSV с теми же картинками/вопросом/порядком.
Запуск: python verify_tsv_export.py [модель-slug ...]   (magma internvla xiaomi gr00t spatialvla)
"""
import csv, glob, json, os, random, re, sys
from collections import defaultdict
import numpy as np

EXP = "/workspace/moskalenko/ws_h100/export_tsv_vla_sim"
REPO = "/workspace/moskalenko/bias-vla-benchmark-main"
MET = REPO + "/metrics"
OUT = REPO + "/Act2Answer/outputs"
ASSETS = REPO + "/Act2Answer/ManiSkill/mani_skill/assets/carrot"
MODELS = {"magma": "microsoft_Magma-8B", "internvla": "InternRobotics_InternVLA-M1",
          "xiaomi": "XiaomiRobotics_Xiaomi-Robotics-0-SimplerEnv-WidowX",
          "gr00t": "nvidia_GR00T-N1.7-SimplerEnv-Bridge", "spatialvla": "IPEC-COMMUNITY_spatialvla-4b-224-pt"}
ETH = ["asian", "black", "latino", "middle_eastern", "white"]
HEADER = ["image_left", "image_right", "question", "answer_side", "answer_soft", "answer_hard", "y_release_mm"]
PROBLEMS = []


def bad(msg):
    PROBLEMS.append(msg)
    print("   ❌", msg)


def toks(part):
    return [t for t in re.split(r"[_.\-]", part.lower()) if t]


def own_parts(path):
    """имя файла, затем папка картинки: группа/пол ищутся в самой картинке, а не в папке сцены
    (в PAIRS есть сцены black_apron, white_shirt — по всему пути они давали ложную группу)"""
    p = path.split("/")
    return [p[-1]] + ([p[-2]] if len(p) > 1 else [])


def gender_of(path):
    for part in own_parts(path):
        tk = toks(part)
        if any(t in ("woman", "female") for t in tk):
            return "female"
        if any(re.fullmatch(r"(man|male)\d*", t) for t in tk):
            return "male"
    return None


def eth_of(path):
    for part in own_parts(path):
        tk = toks(part)
        if "middle" in tk or "me" in tk:
            return "middle_eastern"
        for e in ("asian", "black", "latino", "white"):
            if e in tk:
                return e
    return None


def risk_of(path):
    tk = toks(path)
    return "danger" if "danger" in tk else ("safe" if "safe" in tk else None)


def file_attr(name):
    """-> (функция признака картинки, значение признака файла)"""
    if name == "gender":
        return gender_of, "male"
    if name == "answers":
        return risk_of, "danger"
    m = re.fullmatch(r"(ethnicity|skin_color)_(.+)", name)
    return eth_of, m.group(2)


def read(path):
    with open(path, encoding="utf-8") as fh:
        r = csv.reader(fh, delimiter="\t")
        head = next(r)
        return head, list(r)


# вопрос -> (question_id, axis, polarity) из определений кардсетов
QMAP = {}
for cs in ("focus_g10", "visbias_g10", "pairs_g10", "focus_e10", "visbias_e10", "pairs_e10",
           "focus_e10b", "visbias_e10b", "veri_emerg"):
    d = os.path.join(ASSETS, cs)
    if not os.path.isdir(d):
        continue
    pj = json.load(open(os.path.join(d, "pairs.json"), encoding="utf-8"))
    for r in csv.DictReader(open(os.path.join(d, "episodes.csv"), encoding="utf-8")):
        QMAP[pj[int(r["index"])]["question"]] = (r["question_id"], r["axis"], r["polarity"])


def side_y(ystr):
    return "L" if ystr.startswith("-") else "R"


def check_file(path, rows_by_file):
    name = os.path.basename(path)[:-4]
    f_of, want = file_attr(name)
    head, rows = read(path)
    if head != HEADER:
        bad(f"{path}: заголовок {head}")
    n_side = n_hs = n_pair = n_swap = 0
    for i, r in enumerate(rows):
        L, R, q, a_side, a_soft, a_hard, y = r
        aL, aR = f_of(L), f_of(R)
        if aL is None or aR is None:
            bad(f"{path}:{i+2}: не разобрал имя {L} / {R}")
            continue
        if (aL == want) + (aR == want) != 1:
            n_pair += 1
            continue
        chosen = aL if side_y(y) == "L" else aR
        if (chosen == want) != (a_side == "True"):
            n_side += 1
        if a_hard != "" and a_soft != a_hard:
            n_hs += 1
        if a_side not in ("True", "False") or a_soft not in ("True", "False", "") or a_hard not in ("True", "False", ""):
            bad(f"{path}:{i+2}: значения {a_side}/{a_soft}/{a_hard}")
        if i % 2 == 1:
            p = rows[i - 1]
            if not (p[0] == R and p[1] == L and p[2] == q):
                n_swap += 1
    rows_by_file[name] = rows
    for cnt, what in ((n_side, "answer_side ≠ признак картинки на стороне знака y"),
                      (n_pair, "признак файла не ровно у одной картинки"),
                      (n_hs, "hard задан, а soft с ним не совпадает"),
                      (n_swap, "swap-строка ≠ переставленная noswap")):
        if cnt:
            bad(f"{path}: {cnt} строк: {what}")
    return len(rows)


def sym_check(label, rows_by_file, prefix):
    """пара в <prefix><v> и <prefix><u> с обратным answer_side"""
    idx = {}
    for name, rows in rows_by_file.items():
        if name.startswith(prefix):
            for r in rows:
                idx[(name, r[0], r[1], r[2], r[6])] = r[3]
    miss = 0
    for (name, L, R, q, y), a in idx.items():
        g = name[len(prefix):]
        other = eth_of(R) if eth_of(L) == g else eth_of(L)
        b = idx.get((prefix + other, L, R, q, y))
        if b is None or b == a:
            miss += 1
    if miss:
        bad(f"{label} {prefix}*: {miss} строк без зеркальной пары в файле другой группы")
    else:
        print(f"   ✅ {label} {prefix}*: все {len(idx)} строк имеют зеркальную пару с обратным ответом")


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def compare(label, tsv_vals, csv_path, key_cols, val_col="d"):
    """tsv_vals: {(topic, demo): value}; сравнение с pull_release в csv"""
    if not os.path.exists(csv_path):
        print(f"   · нет {os.path.basename(csv_path)} — пропуск")
        return
    ref = {}
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        if r["metric"] == "pull_release":
            ref[tuple(r[c] for c in key_cols)] = float(r[val_col])
    worst, nmatch = 0.0, 0
    for k, v in tsv_vals.items():
        if k not in ref:
            continue
        nmatch += 1
        worst = max(worst, abs(v - ref[k]))
    missing = [k for k in ref if k not in tsv_vals]
    ok = nmatch and worst < 0.2 and not missing
    print(f"   {'✅' if ok else '❌'} {label}: {nmatch} ячеек против {os.path.basename(csv_path)}, "
          f"макс |TSV − метрики| = {worst:.3f} мм" + (f", нет в TSV: {missing[:4]}" if missing else ""))
    if not ok:
        PROBLEMS.append(f"{label}: расхождение с {csv_path} ({worst:.3f} мм, пропущено {len(missing)})")
    shown = 0
    for k in sorted(tsv_vals):
        if k in ref and shown < 2:
            print(f"      пример {k}: TSV {tsv_vals[k]:+.2f}  метрики {ref[k]:+.2f}")
            shown += 1


def toward(rows, f_of, target):
    """(вопрос, y к картинке с признаком target в мм, левая, правая)"""
    for r in rows:
        L, R, q, y = r[0], r[1], r[2], float(r[6])
        yield q, (y if f_of(R) == target else -y), L, R


def delta(acc):
    return {(a, dm): mean(acc[(a, "pos", dm)]) - mean(acc[(a, "neg", dm)])
            for (a, p, dm) in list(acc) if p == "pos" and (a, "neg", dm) in acc}


def aggregates(vla, rbf_by_ds):
    for ds in ("focus", "visbias", "pairs"):
        rbf = rbf_by_ds.get(ds, {})
        # гендер: Δ = pull(pos) − pull(neg), + = к мужчине
        if "gender" in rbf:
            acc = defaultdict(list)
            for q, t, L, R in toward(rbf["gender"], gender_of, "male"):
                qid, axis, pol = QMAP[q]
                acc[(axis, pol, "gender")].append(t)
            compare(f"{vla} {ds} гендер", delta(acc), f"{MET}/g10_{vla}_{ds}_g10.csv", ("topic", "demo"))
        # этничность e10: X→white (+ к белому) и all→white; в PAIRS — skin_color
        wf = "ethnicity_white" if ds != "pairs" else "skin_color_white"
        if wf in rbf:
            acc = defaultdict(list)
            for q, t, L, R in toward(rbf[wf], eth_of, "white"):
                qid, axis, pol = QMAP[q]
                other = eth_of(R) if eth_of(L) == "white" else eth_of(L)
                acc[(axis, pol, f"{other}→white")].append(t)
                acc[(axis, pol, "all→white")].append(t)
            compare(f"{vla} {ds} этничность e10", delta(acc), f"{MET}/e10_{vla}_{ds}_e10.csv", ("topic", "demo"))
        # e10b: a→b, a раньше b в ETH, оба небелые, + к b
        if ds != "pairs":
            acc = defaultdict(list)
            for b in ETH[:-1]:
                fn = f"ethnicity_{b}"
                if fn not in rbf:
                    continue
                for q, t, L, R in toward(rbf[fn], eth_of, b):
                    other = eth_of(R) if eth_of(L) == b else eth_of(L)
                    if other == "white" or ETH.index(other) > ETH.index(b):
                        continue
                    qid, axis, pol = QMAP[q]
                    acc[(axis, pol, f"{other}→{b}")].append(t)
            vals = delta(acc)
            if vals:
                compare(f"{vla} {ds} этничность e10b", vals, f"{MET}/e10b_{vla}_{ds}_e10b.csv", ("topic", "demo"))
    # VERI: pull vs 0 по вопросу, + к безопасной сцене
    rbf = rbf_by_ds.get("EMERGENCY", {})
    if "answers" in rbf:
        acc = defaultdict(list)
        for q, t, L, R in toward(rbf["answers"], risk_of, "safe"):
            acc[(QMAP[q][0], "pairs")].append(t)
        compare(f"{vla} VERI", {k: mean(v) for k, v in acc.items()},
                f"{MET}/veri_{vla}_veri_emerg_byq_abs.csv", ("topic", "demo"), "mean")


def run_prefix(prog, vla, cs):
    return f"g10-{cs}" if (vla == "magma" and prog == "g10") else f"{prog}-{vla}-{cs}"


def raw_spot(vla, rbf_by_ds, k_per=4):
    rnd = random.Random(15)
    runs = [("g10", "focus_g10", "focus"), ("g10", "visbias_g10", "visbias"), ("g10", "pairs_g10", "pairs"),
            ("e10", "focus_e10", "focus"), ("e10", "pairs_e10", "pairs"), ("e10b", "visbias_e10b", "visbias"),
            ("veri", "veri_emerg", "EMERGENCY")]
    ok = tot = 0
    for prog, cs, ds in runs:
        eps = {int(r["index"]): r for r in csv.DictReader(open(os.path.join(ASSETS, cs, "episodes.csv"), encoding="utf-8"))}
        pj = json.load(open(os.path.join(ASSETS, cs, "pairs.json"), encoding="utf-8"))
        for order in ("noswap", "swap"):
            shards = sorted(glob.glob(f"{OUT}/{run_prefix(prog, vla, cs)}-{order}-s*/glob/vis_0_test/traj.npz"))
            if not shards:
                continue
            for f in rnd.sample(shards, min(k_per, len(shards))):
                z = np.load(f, allow_pickle=True)
                k = rnd.randrange(len(z["ep_ids"]))
                ep = int(z["ep_ids"][k])
                g = np.where(z["grasped"][k])[0]
                y = float(z["cube_xyz"][k, int(g[-1]) if len(g) else -1, 1])
                e = eps[ep]
                L, R = e["left_image"], e["right_image"]
                if order == "swap":
                    L, R = R, L
                q = pj[ep]["question"]
                ystr = f"{y * 1000:+.1f}"
                src = e["source"]
                if src == "tsv:gender":
                    fn = "gender"
                elif src == "tsv:pairs":
                    fn = "answers"
                elif src == "tsv:skin_color":
                    fn = f"skin_color_{e['attr_skin_color_1']}"
                else:
                    fn = f"ethnicity_{e['attr_ethnicity_1']}"
                rows = rbf_by_ds.get(ds, {}).get(fn, [])
                hit = any(r[0] == L and r[1] == R and r[2] == q and r[6] == ystr for r in rows)
                tot += 1
                ok += hit
                if not hit:
                    bad(f"{vla} {cs} {order} ep{ep}: из traj.npz y={ystr}, в {ds}/{fn}.tsv такой строки нет")
    print(f"   {'✅' if ok == tot else '❌'} сырьё: {ok}/{tot} случайных эпизодов из traj.npz найдены в TSV с тем же y")


for vla in (sys.argv[1:] or list(MODELS)):
    mname = MODELS[vla]
    print(f"\n=== {vla} ({mname})")
    rbf_by_ds = {}
    for ds in ("focus", "visbias", "pairs", "EMERGENCY"):
        mdir = os.path.join(EXP, ds, mname)
        if not os.path.isdir(mdir):
            print(f"   · {ds}: нет папки")
            continue
        rbf = {}
        counts = {os.path.basename(p)[:-4]: check_file(p, rbf) for p in sorted(glob.glob(mdir + "/*.tsv"))}
        print(f"   {ds}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        if ds in ("focus", "visbias"):
            sym_check(f"{ds}", rbf, "ethnicity_")
        if ds == "pairs":
            sym_check(f"{ds}", rbf, "skin_color_")
        rbf_by_ds[ds] = rbf
    aggregates(vla, rbf_by_ds)
    raw_spot(vla, rbf_by_ds)

print("\nИТОГ:", "✅ проблем нет" if not PROBLEMS else f"❌ {len(PROBLEMS)} проблем")
for p in PROBLEMS[:30]:
    print("  -", p)
