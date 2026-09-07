#!/usr/bin/env python3
"""M2 paper figures. All numbers read from archived JSONs (no recomputation of
RF3 outputs). Sources:
  E2/E49 range-matched : /mnt/k/output_heads/e2/e2_rangematched_results.json
  E49b GA88/GB88       : /mnt/k/output_heads/e2/E2-06/e49b_results.json
  E3 paired            : /mnt/k/output_heads/casp3/e3_blocks/results.json
  E3 unpaired          : /mnt/k/output_heads/casp3/e3_blocks/zii_msa_unpaired/results.json
  E49-RBD geometry     : /mnt/k/output_heads/rbd/e1_msa_arms_analysis/results_e49.json
  E49-RBD supervised   : /mnt/k/output_heads/rbd/e1_supervised/results_e49.json
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/mnt/j/conda_envs/foundry/DMS_Project/sidepapers/m2_state_selection/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# Tol palette
C_TOWARD = "#CC6677"; C_AWAY = "#4477AA"; C_NULL = "#BBBBBB"; C_SHARP = "#66CCEE"
C_SEQ = "#999999"; C_MSA = "#4477AA"; C_TPL = "#CC6677"; C_BOTH = "#AA3377"

# ---------------------------------------------------------------- fig 1: E2
rm = {r["id"]: r for r in json.load(open("/mnt/k/output_heads/e2/e2_rangematched_results.json"))}
# crystallized state is A for all except Mad2 (crystal = C-Mad2 = state B)
SIDE_B = {"E2-03"}
rows = []
for tid, r in rm.items():
    s = r["shift"]
    toward = -s if tid in SIDE_B else s
    rows.append((tid, r["name"], toward, r["category"]))
rows.sort(key=lambda x: -x[2])

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.axvspan(-0.27, 0.27, color="0.92", zorder=0)
ax.axvspan(-0.14, 0.14, color="0.85", zorder=0)
ax.axvline(0, color="k", lw=0.8)
labels = []
for i, (tid, name, toward, cat) in enumerate(rows):
    if "GA88" in name:
        c = C_SHARP; h = "//"; lab_suffix = " (own-fold sharpen.)"
    elif abs(toward) <= 0.27:
        c = C_NULL; h = None; lab_suffix = ""
    elif toward > 0:
        c = C_TOWARD; h = None; lab_suffix = ""
    else:
        c = C_AWAY; h = None; lab_suffix = ""
    ax.barh(i, toward, color=c, hatch=h, edgecolor="white", height=0.75, zorder=3)
    ax.text(toward + (0.06 if toward >= 0 else -0.06), i, f"{toward:+.2f}",
            va="center", ha="left" if toward >= 0 else "right", fontsize=8, zorder=4)
    short = {"Mad2 (MAD2A)": "Mad2", "XCL1 / lymphotactin": "XCL1",
             "Selecase (MJ1213)": "selecase", "GA88 / GB88 (designed)": "GA88/GB88",
             "beta2-adrenergic receptor (beta2AR)": r"$\beta$2AR",
             "LacY (lactose permease)": "LacY", "GB1 (protein G B1 domain)": "GB1",
             "Thioredoxin (TrxA)": "thioredoxin"}.get(name, name)
    labels.append(short + lab_suffix)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlim(-3.9, 3.3)
ax.set_xlabel("MSA-induced shift toward the crystallized state\n"
              r"(pref$_{\mathrm{msa}}-$pref$_{\mathrm{nomsa}}$, sign-corrected per target; >0 = toward)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=C_TOWARD, label="toward crystal"),
                   Patch(fc=C_AWAY, label="away from crystal"),
                   Patch(fc=C_NULL, label="within noise band"),
                   Patch(fc=C_SHARP, hatch="//", label="own-fold sharpening")],
          loc="lower right", fontsize=7.5, frameon=False)
ax.text(0.02, 0.02, "grey bands: run-to-run noise $\\pm$0.14 / $\\pm$0.27", transform=ax.transAxes, fontsize=7.5)
fig.savefig(OUT / "fig1_e2_shifts.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 2: E49
tids49 = ["E2-01", "E2-02", "E2-03", "E2-04", "E2-10"]
names49 = ["KaiB", "RfaH", "Mad2", "XCL1", "LacY"]
arms = ["nomsa", "msa", "tplA", "tplB", "msa_tplA"]
arm_lab = ["no MSA", "MSA", "template A", "template B", "MSA + template A"]
arm_col = [C_SEQ, C_MSA, "#DD8899", "#882255", C_BOTH]
x = np.arange(len(tids49)); w = 0.16
fig, ax = plt.subplots(figsize=(6.8, 3.4))
for j, (a, l, c) in enumerate(zip(arms, arm_lab, arm_col)):
    vals = [rm[t][f"pref_{a}"] for t in tids49]
    ax.bar(x + (j - 2) * w, vals, width=w * 0.92, color=c, label=l)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(names49)
ax.set_ylabel(r"state preference  pref = NLL$_B-$NLL$_A$")
ax.legend(fontsize=7.5, frameon=False, ncol=5, loc="upper left", bbox_to_anchor=(0, 1.14))
ax.set_title(">0 favours state A (crystallized state for KaiB/RfaH/XCL1/LacY; O-state for Mad2)", fontsize=8)
fig.savefig(OUT / "fig2_e49_arms.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 3: E49b
b = json.load(open("/mnt/k/output_heads/e2/E2-06/e49b_results.json"))
arms3 = ["nomsa", "msa", "tplOwn", "tplOther", "msa_tplOther"]
lab3 = ["no MSA\n(seq. only)", "MSA", "template\nown fold", "template\nother fold", "MSA + template\nother fold"]
x = np.arange(len(arms3)); w = 0.36
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.bar(x - w / 2, [b["GA88"][a] for a in arms3], width=w * 0.9, color="#228833", label="GA88")
ax.bar(x + w / 2, [b["GB88"][a] for a in arms3], width=w * 0.9, color="#EE6677", label="GB88")
for i, a in enumerate(arms3):
    for off, k in [(-w / 2, "GA88"), (w / 2, "GB88")]:
        v = b[k][a]
        ax.text(i + off, v + (0.15 if v >= 0 else -0.15), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(lab3, fontsize=8)
ax.set_ylabel(r"pref = NLL$_{GB88\ fold}-$NLL$_{GA88\ fold}$")
ax.legend(fontsize=8, frameon=False)
fig.savefig(OUT / "fig3_e49b_gagb.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 4: E3
e3p = json.load(open("/mnt/k/output_heads/casp3/e3_blocks/results.json"))["summary"]
e3u = json.load(open("/mnt/k/output_heads/casp3/e3_blocks/zii_msa_unpaired/results.json"))["summary"]
blocks = ["AA", "BB", "AB"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.9))
x = np.arange(3); w = 0.36
a1.bar(x - w / 2, [e3p[f"cos_{k}"]["median"] for k in blocks], w * 0.9, color="#4477AA", label="paired MSA (n=392)")
a1.bar(x + w / 2, [e3u[f"cos_{k}"]["median"] for k in blocks], w * 0.9, color="#CCBB44", label="unpaired control (n=100)")
a1.set_xticks(x); a1.set_xticklabels(["A–A", "B–B", "A–B"])
a1.set_ylabel(r"median cos($Z^{msa}$, $Z^{no}$)"); a1.set_ylim(0.68, 0.78)
a1.legend(fontsize=7.5, frameon=False, loc="lower left")
a1.set_title("direction preservation", fontsize=9)
a2.bar(x - w / 2, [e3p[f"normratio_{k}"]["median"] for k in blocks], w * 0.9, color="#4477AA", label="paired")
a2.bar(x + w / 2, [e3u[f"normratio_{k}"]["median"] for k in blocks], w * 0.9, color="#CCBB44", label="unpaired")
a2.set_xticks(x); a2.set_xticklabels(["A–A", "B–B", "A–B"])
a2.set_ylabel(r"median $||Z^{msa}||\,/\,||Z^{no}||$"); a2.set_ylim(1.0, 1.35)
a2.axhline(1, color="k", lw=0.6, ls=":")
a2.set_title("amplitude change", fontsize=9)
fig.savefig(OUT / "fig4_e3_blocks.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 5: E49-RBD
geo = json.load(open("/mnt/k/output_heads/rbd/e1_msa_arms_analysis/results_e49.json"))["arms"]
sup = json.load(open("/mnt/k/output_heads/rbd/e1_supervised/results_e49.json"))
supmap = {"no": "no MSA", "zii_e1_full15": "MSA (15 rows)", "zii_e49_tpl": "template (6M0J)", "zii_e49_msatpl": "MSA + template"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
conds = ["no", "zii_e1_full15", "zii_e49_tpl", "zii_e49_msatpl"]
cols = [C_SEQ, C_MSA, C_TPL, C_BOTH]
for i, (c, col) in enumerate(zip(conds, cols)):
    vals = [r["val"] for r in sup["runs"] if r["cond"] == c]
    m = sup["summary"][c]["mean"]
    a1.bar(i, m, color=col, width=0.7, zorder=2)
    a1.scatter(np.full(len(vals), i) + np.linspace(-0.22, 0.22, len(vals)), vals,
               s=6, color="k", alpha=0.55, zorder=3)
    a1.text(i, 0.205, f"{m:.3f}", ha="center", fontsize=8.5, color="white", weight="bold")
a1.set_xticks(range(4)); a1.set_xticklabels(["no MSA", "MSA\n(15 rows)", "template\n(6M0J)", "MSA +\ntemplate"], fontsize=8)
a1.set_ylabel(r"cross-position Spearman $\rho$ (val)")
a1.set_ylim(0.15, 0.74); a1.set_title("(a) mutation-effect prediction\n(RBD DMS, 500-subset, 5$\\times$3)", fontsize=8.5)
arms5 = ["origMSA_ref", "zii_e49_tpl", "zii_e49_msatpl"]
metrics = [("cka_vs_noMSA", "CKA vs no-MSA"),
           ("delta_cos_vs_noMSA_median", "mutation-$\\Delta$ direction (median cos)"),
           ("delta_norm_ratio_median", "mutation-$\\Delta$ norm ratio (median)")]
x = np.arange(len(metrics)); w = 0.26
for j, (a, col) in enumerate(zip(arms5, cols[1:])):
    vals = [geo[a][m] for m, _ in metrics]
    a2.bar(x + (j - 1) * w, vals, width=w * 0.92, color=col,
           label=["MSA (15 rows)", "template", "MSA + template"][j])
    for xi, v in zip(x + (j - 1) * w, vals):
        a2.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
a2.set_xticks(x)
a2.set_xticklabels(["CKA\nvs no-MSA", "$\\Delta$ dir.\ncos", "$\\Delta$ norm\nratio"], fontsize=8)
a2.axhline(1, color="k", lw=0.6, ls=":")
a2.set_ylim(0, 1.75)
a2.legend(fontsize=7, frameon=False, loc="upper left")
a2.set_title("(b) representation geometry\n(500-subset; no-MSA eff. rank 181)", fontsize=8.5)
fig.savefig(OUT / "fig5_e49rbd.png"); plt.close(fig)

print("done:", sorted(p.name for p in OUT.glob("*.png")))
