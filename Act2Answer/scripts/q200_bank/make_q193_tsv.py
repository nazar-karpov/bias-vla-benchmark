# -*- coding: utf-8 -*-
"""questions_q193.tsv: банк FairACT (200 вопросов, selection.json из VLA_LINPROBING/q200) минус 7,
уже прогнанных на VLA в той же формулировке (x3 ×3, criminal ×2, education ×2).
6 вопросов g10 (pilot/FA/janitor/SAHP/wealthy/poor) шли в короткой форме «Put cube on the X» —
здесь они в формулировке банка и стоят В КОНЦЕ (наименьший приоритет).
Порядок: круговой обход 7 категорий, полюса одной PAIRS-пары (pole_a/pole_b) рядом —
любой префикс прогона покрывает все категории и обе полярности пар."""
import csv, json, re, collections
S = json.load(open("selection.json", encoding="utf-8"))
DONE_EXACT = {"visbias_form_physical_appearance_muscular", "visbias_mcq_education_doctorate",
              "visbias_form_education_no_education", "visbias_form_disability_status_intellectual_disabilities",
              "visbias_form_hobby_or_interest_sport", "visbias_form_criminal_record_yes", "visbias_form_criminal_record_no"}
G10_SHORT = {"pairs_status_wealthy_person_vs_poor_person_pole_a", "pairs_status_wealthy_person_vs_poor_person_pole_b",
             "pairs_occupations_janitor_vs_stay_at_home_parent_pole_a", "pairs_occupations_janitor_vs_stay_at_home_parent_pole_b",
             "pairs_occupations_pilot_vs_flight_attendant_pole_a", "pairs_occupations_pilot_vs_flight_attendant_pole_b"}
CAT = {"Demographics and Appearance": "C1", "Education and Socioeconomic Status": "C2",
       "Occupations, Employment and Commercial Roles": "C3", "Family, Culture and Civic Context": "C4",
       "Health, Disability and Substance Use": "C5", "Personality and Interests": "C6",
       "Trust, Safety and Legal Status": "C7"}
slug = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
rows = []
for q in S["questions"]:
    qid = q["question_id"]
    if qid in DONE_EXACT:
        continue
    cat = CAT[q["category_heading"].rsplit(" — ", 1)[0]]
    sub = q["subcategory_heading"].rsplit(" (", 1)[0]
    m = re.match(r"pairs_(\w+?)_(.+)_pole_([ab])$", qid)
    if m:   # PAIRS: ось = пара полюсов, pos = pole_a, neg = pole_b (полярный контроль)
        axis, pol = m.group(2), ("pos" if m.group(3) == "a" else "neg")
    else:
        axis, pol = slug(sub), ""
    rows.append(dict(question_id=qid, stable_id=q["stable_id"], category=cat, subcategory=sub,
                     attribute=axis, polarity=pol, question_vla=q["question_vla"]))
assert len(rows) == 193, len(rows)
# единицы очереди: пара полюсов = одна единица
units = collections.OrderedDict()
for r in rows:
    key = (r["category"], r["attribute"]) if r["polarity"] else (r["category"], r["question_id"])
    units.setdefault(key, []).append(r)
late = [u for u in units.values() if any(r["question_id"] in G10_SHORT for r in u)]
main = [u for u in units.values() if u not in late]
bycat = collections.OrderedDict((c, []) for c in sorted(set(CAT.values())))
for u in main:
    bycat[u[0]["category"]].append(u)
order = []
while any(bycat.values()):
    for c in bycat:
        if bycat[c]:
            order.append(bycat[c].pop(0))
order += late
out = [r for u in order for r in sorted(u, key=lambda r: r["polarity"] != "pos")]
assert len(out) == 193 and len({r["question_id"] for r in out}) == 193
with open("questions_q193.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()), delimiter="\t", lineterminator="\n")
    w.writeheader(); w.writerows(out)
print("rows", len(out), collections.Counter(r["category"] for r in out))
print("pole pairs:", sum(1 for r in out if r["polarity"] == "pos"), "pos /", sum(1 for r in out if r["polarity"] == "neg"), "neg")
print("first 14:", [r["stable_id"] for r in out[:14]])
print("last 8:", [r["stable_id"] for r in out[-8:]])
