"""Perturbation ablation.

For each of Candidate A and Candidate B, perturb every dimension by +0.05
and by -0.05 (clipped to [0, 1]) one at a time, re-rank the seven
occupations, and count how often the top-3 changes.

This tests whether the rankings reported in the Illustrative Example are
stable under small changes in the candidate's self-assessment values --
a natural robustness concern for a recommender whose input is a
self-report on a small number of dimensions.
"""

import numpy as np
from recommender import OCCUPATIONS, DEMAND, cosine_similarity as cosine_sim

DIMS = [
    "Analytical", "Creative", "Systems", "Resilience",
    "Empathy & active listening", "Oral communication",
    "Written communication", "Leadership & social influence",
    "Service orientation", "Technological literacy",
]

CANDIDATE_A = np.array([0.85, 0.55, 0.75, 0.50, 0.55, 0.45, 0.80, 0.35, 0.45, 0.80])
CANDIDATE_B = np.array([0.45, 0.55, 0.40, 0.85, 0.85, 0.30, 0.55, 0.45, 0.85, 0.45])


def rank(candidate):
    scores = {name: cosine_sim(candidate, np.array(vec))
              for name, vec in OCCUPATIONS.items()}
    return sorted(scores.items(), key=lambda kv: -kv[1])


def top_k(ranking, k=3):
    return tuple(name for name, _ in ranking[:k])


def perturbation_check(name, candidate, delta=0.05):
    base = rank(candidate)
    base_top1 = base[0][0]
    base_top3 = top_k(base, 3)
    print(f"\n=== {name} ===")
    print(f"Baseline top-1: {base_top1}")
    print(f"Baseline top-3: {base_top3}")
    print(f"\nPerturbing each dimension by \u00b1{delta} (clipped to [0,1]):\n")
    print(f"{'Dimension':<32}{'Delta':<8}{'Top-1':<28}{'Top-3 changed?':<15}")
    print("-" * 85)
    top1_changes = 0
    top3_changes = 0
    total = 0
    for i, dim in enumerate(DIMS):
        for sign in [+1, -1]:
            perturbed = candidate.copy()
            perturbed[i] = float(np.clip(candidate[i] + sign * delta, 0.0, 1.0))
            r = rank(perturbed)
            new_top1 = r[0][0]
            new_top3 = top_k(r, 3)
            top1_ch = new_top1 != base_top1
            top3_ch = new_top3 != base_top3
            if top1_ch:
                top1_changes += 1
            if top3_ch:
                top3_changes += 1
            total += 1
            flag = "yes" if top3_ch else "no"
            print(f"{dim:<32}{sign*delta:<+8.2f}{new_top1:<28}{flag:<15}")
    print(f"\nTotal perturbations tested: {total}")
    print(f"Perturbations that changed top-1: {top1_changes} / {total}")
    print(f"Perturbations that changed top-3 set: {top3_changes} / {total}")


if __name__ == "__main__":
    perturbation_check("Candidate A", CANDIDATE_A)
    perturbation_check("Candidate B (Deaf, general labour market)", CANDIDATE_B)


# Sunyi-inclusive scenario for Candidate B: oral communication rescaled to 0.75
CANDIDATE_B_SUNYI = CANDIDATE_B.copy()
CANDIDATE_B_SUNYI[5] = 0.75

if __name__ == "__main__":
    print("\n\n" + "=" * 85)
    print("Extra: Candidate B in the Sunyi-inclusive scenario")
    print("=" * 85)
    perturbation_check("Candidate B (Sunyi-inclusive scenario)", CANDIDATE_B_SUNYI)
