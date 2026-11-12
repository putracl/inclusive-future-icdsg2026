"""
Figure 1 for the paper. Uses the canonical vectors from recommender.py
and the canonical Candidate B (Deaf) from candidate_b_deaf.py.

Two panels:
  (a) Hearing-default workplace: oral comm = 0.30
  (b) Deaf-inclusive workplace (Sunyi/Difabis): oral comm rescaled to 0.75

Each panel shows the top-3 ranked occupations with per-dimension contribution
stacked into the bar. Cosine similarity value labelled at bar end.
"""
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from recommender import OCCUPATIONS, DIMENSIONS
from candidate_b_deaf import CANDIDATE_B_DEAF, CANDIDATE_B_SUNYI

# Shorten dimension names for legend
DIM_SHORT = [
    "Analytical", "Creative", "Systems", "Resilience",
    "Empathy", "Oral comm", "Written comm", "Leadership",
    "Service", "Tech literacy",
]


def l2(v):
    m = math.sqrt(sum(x * x for x in v))
    return v if m == 0 else [x / m for x in v]


def cosine(a, b):
    an, bn = l2(a), l2(b)
    return sum(x * y for x, y in zip(an, bn))


def per_dim_contribution(cand, occ):
    cn = l2(cand)
    on = l2(occ)
    return [c * o for c, o in zip(cn, on)]


def rank(cand):
    scored = [(name, cosine(cand, v)) for name, v in OCCUPATIONS.items()]
    return sorted(scored, key=lambda x: -x[1])


def summarise(cand, label):
    ranking = rank(cand)
    top3 = ranking[:3]
    print(f"\n{label}")
    for name, score in ranking:
        print(f"  {name:28s} {score:.4f}")
    return top3


hearing_top3 = summarise(CANDIDATE_B_DEAF, "Hearing-default workplace (oral comm = 0.30)")
inclusive_top3 = summarise(CANDIDATE_B_SUNYI, "Deaf-inclusive (Sunyi) workplace (oral comm = 0.75)")


# CSV
csv_rows = [["scoring", "occupation", "cosine"] + DIM_SHORT]
for name, score in hearing_top3:
    contrib = per_dim_contribution(CANDIDATE_B_DEAF, OCCUPATIONS[name])
    csv_rows.append(["hearing_default", name, f"{score:.4f}"] + [f"{v:.4f}" for v in contrib])
for name, score in inclusive_top3:
    contrib = per_dim_contribution(CANDIDATE_B_SUNYI, OCCUPATIONS[name])
    csv_rows.append(["deaf_inclusive", name, f"{score:.4f}"] + [f"{v:.4f}" for v in contrib])

# Use script directory so this runs on any machine
SCRIPT_DIR = Path(__file__).resolve().parent
csv_path = SCRIPT_DIR / "figure1_data.csv"
with csv_path.open("w", newline="") as f:
    csv.writer(f).writerows(csv_rows)
print(f"\nData written to {csv_path}")


# --- Figure ---
plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "pdf.fonttype": 42,  # Embed as TrueType (Type 42), not Type 3 (IEEE PDF eXpress requirement)
    "ps.fonttype": 42,
})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharex=True)
colors = plt.cm.tab10(np.linspace(0, 1, 10))


def draw_panel(ax, top3, cand, title):
    names = [t[0] for t in top3]
    y_pos = np.arange(len(names))
    left = np.zeros(len(names))
    for d_idx, dim in enumerate(DIM_SHORT):
        contribs = np.array([
            per_dim_contribution(cand, OCCUPATIONS[name])[d_idx]
            for name in names
        ])
        ax.barh(y_pos, contribs, left=left, color=colors[d_idx],
                edgecolor="white", linewidth=0.3,
                label=dim if title.startswith("(a)") else None)
        left = left + contribs
    for i, (name, score) in enumerate(top3):
        ax.text(left[i] + 0.008, i, f"{score:.3f}", va="center", fontsize=7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Cosine similarity, stacked by dimension", fontsize=7)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.6)


draw_panel(axes[0], hearing_top3, CANDIDATE_B_DEAF,
           "(a) Hearing-default scoring\n     oral comm = 0.30")
draw_panel(axes[1], inclusive_top3, CANDIDATE_B_SUNYI,
           "(b) Deaf-inclusive scoring\n     oral comm = 0.75")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.08, 1, 1])

out_pdf = SCRIPT_DIR / "figure1.pdf"
out_png = SCRIPT_DIR / "figure1.png"
plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
plt.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Figure written to {out_pdf} and {out_png}")
