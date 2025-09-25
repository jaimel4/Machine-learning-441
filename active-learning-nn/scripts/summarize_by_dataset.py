import os, re, glob, json, numpy as np
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

RESULTS_DIR = "experiments/results"
STRATS = {"passive","uncertainty","sensitivity"}

def parse_entry(path):
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"_s(\d+)$", base)
    seed = int(m.group(1)) if m else None
    name = re.sub(r"_s\d+$","", base)
    toks = name.split("_")
    if len(toks) >= 2 and toks[-2:] == ["uncertainty","mc"]:
        strat = "uncertainty"; dataset = "_".join(toks[:-2])
    else:
        strat = toks[-1] if toks[-1] in STRATS else "unknown"
        dataset = "_".join(toks[:-1]) if strat != "unknown" else name
    task = "classification" if dataset.startswith("cls_") else "regression"
    return dataset, strat, task, seed

def safe_load(path):
    try:
        t = open(path).read().strip()
        if not t: return None
        d = json.loads(t)
        return d if isinstance(d, list) else None
    except Exception:
        return None

def final_metrics(logs, task):
    xs = [r["labeled"] for r in logs]
    if task == "classification":
        ys = [r["acc"] for r in logs]; final = ys[-1]; au = np.trapezoid(ys, xs); metric = "acc"
    else:
        rmses = [float(r["loss"])**0.5 for r in logs]; ys = [-rm for rm in rmses]
        final = rmses[-1]; au = np.trapezoid(ys, xs); metric = "rmse"
    return final, au, metric

def ci95(vals):
    arr = np.asarray(list(vals), float)
    if arr.size == 0: return np.nan, np.nan
    m = np.nanmean(arr); s = np.nanstd(arr, ddof=1) if arr.size>1 else 0.0
    return m, 1.96 * s / max(np.sqrt(arr.size),1)

def paired_p(a_dict, b_dict):
    if wilcoxon is None: return None
    # only common **int seeds**; drop None
    keys = sorted(k for k in (set(a_dict) & set(b_dict)) if isinstance(k, int))
    if len(keys) < 5: return None
    a = [a_dict[k] for k in keys]; b = [b_dict[k] for k in keys]
    try:
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return None

def fmt_p(x):
    if x is None: return "-"
    try:
        if x != x: return "-"   # NaN
        return f"{x:.3g}"
    except Exception:
        return "-"

def main():
    groups = {}  # (dataset, strat, task, metric) -> seed -> (final, au)
    for p in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        logs = safe_load(p)
        if logs is None: continue
        dataset, strat, task, seed = parse_entry(p)
        if strat == "unknown": continue
        final, au, metric = final_metrics(logs, task)
        key = (dataset, strat, task, metric)
        groups.setdefault(key, {})
        groups[key][seed] = (final, au)

    datasets = sorted({k[0] for k in groups})
    rows = []
    for d in datasets:
        any_key = next(k for k in groups if k[0]==d)
        task, metric = any_key[2], any_key[3]
        strat_data = {}
        for strat in ("passive","uncertainty","sensitivity"):
            key = (d, strat, task, metric)
            if key in groups:
                finals = {s:v[0] for s,v in groups[key].items()}
                aucs   = {s:v[1] for s,v in groups[key].items()}
                strat_data[strat] = (finals, aucs)
        summary = {}
        for strat, (finals, aucs) in strat_data.items():
            fm, fci = ci95(finals.values()); am, aci = ci95(aucs.values())
            summary[strat]=(fm,fci,am,aci)
        pvals={}
        if "passive" in strat_data:
            pf, pa = strat_data["passive"]
            for comp in ("uncertainty","sensitivity"):
                if comp in strat_data:
                    cf, ca = strat_data[comp]
                    pvals[(comp,"final")] = paired_p(cf, pf)
                    pvals[(comp,"auclc")] = paired_p(ca, pa)
        rows.append((d, task, metric, summary, pvals))

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/summary.csv","w") as f:
        f.write("dataset,task,metric,strategy,final_mean,final_ci95,auclc_mean,auclc_ci95\n")
        for d, task, metric, summary, _ in rows:
            for strat,(fm,fci,am,aci) in summary.items():
                f.write(f"{d},{task},{metric},{strat},{fm:.6f},{fci:.6f},{am:.6f},{aci:.6f}\n")
    print("Wrote experiments/results/summary.csv")

    os.makedirs("report", exist_ok=True)
    with open("report/summary_table.tex","w") as f:
        f.write("\\begin{tabular}{llllrrrr}\n\\toprule\n")
        f.write("Dataset & Task & Metric & Strategy & Final $\\mu$ & $\\pm$CI & AUCLC $\\mu$ & $\\pm$CI \\\\\n\\midrule\n")
        for d, task, metric, summary, pvals in rows:
            for strat,(fm,fci,am,aci) in summary.items():
                f.write(f"{d} & {task} & {metric} & {strat} & {fm:.4f} & {fci:.4f} & {am:.2f} & {aci:.2f} \\\\\n")
            if pvals:  # only print a Wilcoxon line if we have at least one value
                up, ua = fmt_p(pvals.get(("uncertainty","final"))), fmt_p(pvals.get(("uncertainty","auclc")))
                sp, sa = fmt_p(pvals.get(("sensitivity","final"))), fmt_p(pvals.get(("sensitivity","auclc")))
                f.write(f"\\multicolumn{{8}}{{l}}{{\\footnotesize Wilcoxon vs passive — Unc: final p={up}, AUCLC p={ua}; Sen: final p={sp}, AUCLC p={sa}}}\\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("Wrote report/summary_table.tex")

if __name__ == "__main__":
    main()
