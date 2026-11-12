"""
Robustness checks for the O*NET-anchored recommender.

Three simple checks:
1. Mapping sensitivity: max vs average aggregation for the dimensions
2. Null baseline: what does cosine similarity look like against random candidates
3. Content Creator adjustment: with vs without the 0.75 override

All checks report whether Candidate A and B rankings change.
"""

import math
import random
from compute_vectors import ONET_DATA, MAPPING, RESILIENCE, OCCUPATION_ORDER, ORDER

random.seed(42)

CANDIDATE_A = [0.85, 0.55, 0.75, 0.50, 0.55, 0.45, 0.80, 0.35, 0.45, 0.80]
# Candidate B (Deaf, hearing-default scoring) as used in the paper's
# Illustrative Example. Order matches ORDER in compute_vectors:
# [analytical, creative, systems, empathy, service, oral, written,
#  technological, resilience, leadership].
CANDIDATE_B = [0.45, 0.55, 0.40, 0.85, 0.85, 0.30, 0.55, 0.45, 0.85, 0.45]


def l2(v):
    m = math.sqrt(sum(x * x for x in v))
    return v if m == 0 else [x / m for x in v]


def cosine(a, b):
    an, bn = l2(a), l2(b)
    return sum(x * y for x, y in zip(an, bn))


def build_vectors(aggregator, apply_content_adjustment):
    """Build the seven occupation vectors given an aggregator (max or avg)
    and whether to apply the Content Creator Creative adjustment."""
    vectors = {}
    for occ in OCCUPATION_ORDER:
        onet = ONET_DATA[occ]
        vec = {}
        for dim, skills in MAPPING.items():
            vals = [onet.get(s, 0) for s in skills]
            if not vals:
                vec[dim] = 0.0
            elif aggregator == "max":
                vec[dim] = max(vals) / 100.0
            elif aggregator == "avg":
                vec[dim] = (sum(vals) / len(vals)) / 100.0
            else:
                raise ValueError(aggregator)
        vec["Resilience"] = RESILIENCE[occ]
        if apply_content_adjustment and occ == "Content Creator":
            vec["Creative"] = 0.75
        vectors[occ] = [round(vec[d], 3) for d in ORDER]
    return vectors


def ranking(candidate, vectors):
    scored = [(occ, cosine(candidate, v)) for occ, v in vectors.items()]
    scored.sort(key=lambda x: -x[1])
    return scored


def rank_string(scored):
    return " > ".join(f"{o.split()[0]}({s:.3f})" for o, s in scored)


def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --- Baseline (current design: max, with adjustment) ---
baseline = build_vectors("max", apply_content_adjustment=True)
baseline_a = ranking(CANDIDATE_A, baseline)
baseline_b = ranking(CANDIDATE_B, baseline)

print_section("BASELINE: max aggregator, Content Creator Creative = 0.75")
print("Candidate A:", rank_string(baseline_a))
print("Candidate B:", rank_string(baseline_b))


# --- Check 1: Mapping sensitivity - average instead of max ---
print_section("CHECK 1: Average aggregator instead of max (with adjustment)")
avg_vectors = build_vectors("avg", apply_content_adjustment=True)
avg_a = ranking(CANDIDATE_A, avg_vectors)
avg_b = ranking(CANDIDATE_B, avg_vectors)
print("Candidate A:", rank_string(avg_a))
print("Candidate B:", rank_string(avg_b))

top3_stable_a = [o for o, _ in baseline_a[:3]] == [o for o, _ in avg_a[:3]]
top3_stable_b = [o for o, _ in baseline_b[:3]] == [o for o, _ in avg_b[:3]]
top1_stable_a = baseline_a[0][0] == avg_a[0][0]
top1_stable_b = baseline_b[0][0] == avg_b[0][0]

print()
print(f"Top-1 stable: A={top1_stable_a}, B={top1_stable_b}")
print(f"Top-3 (as set) stable: A={top3_stable_a}, B={top3_stable_b}")


# --- Check 2: Null baseline with random candidates ---
print_section("CHECK 2: Null baseline - random candidate similarity distribution")
N = 1000
random_sims = {occ: [] for occ in OCCUPATION_ORDER}
for _ in range(N):
    rand_cand = [random.random() for _ in range(10)]
    for occ, vec in baseline.items():
        random_sims[occ].append(cosine(rand_cand, vec))

print(f"Random candidates: N={N}, uniform [0,1] on each dimension")
print(f"{'Occupation':<28}{'Mean':<10}{'Std':<10}{'Min':<10}{'Max':<10}")
print("-" * 68)
for occ in OCCUPATION_ORDER:
    sims = random_sims[occ]
    mean = sum(sims) / len(sims)
    var = sum((s - mean) ** 2 for s in sims) / len(sims)
    std = math.sqrt(var)
    print(f"{occ:<28}{mean:<10.3f}{std:<10.3f}{min(sims):<10.3f}{max(sims):<10.3f}")

print()
print("Candidate A scores vs random-candidate mean (in std units):")
for occ, score in baseline_a:
    sims = random_sims[occ]
    mean = sum(sims) / len(sims)
    var = sum((s - mean) ** 2 for s in sims) / len(sims)
    std = math.sqrt(var) if var > 0 else 1
    z = (score - mean) / std
    print(f"  {occ:<28}score={score:.3f}  random_mean={mean:.3f}  z={z:+.2f}")

print()
print("Candidate B scores vs random-candidate mean (in std units):")
for occ, score in baseline_b:
    sims = random_sims[occ]
    mean = sum(sims) / len(sims)
    var = sum((s - mean) ** 2 for s in sims) / len(sims)
    std = math.sqrt(var) if var > 0 else 1
    z = (score - mean) / std
    print(f"  {occ:<28}score={score:.3f}  random_mean={mean:.3f}  z={z:+.2f}")


# --- Check 3: With vs without Content Creator adjustment ---
print_section("CHECK 3: Content Creator adjustment - with vs without")
noadj = build_vectors("max", apply_content_adjustment=False)
noadj_a = ranking(CANDIDATE_A, noadj)
noadj_b = ranking(CANDIDATE_B, noadj)
print("With adjustment (0.75):")
print("  Candidate A:", rank_string(baseline_a))
print("  Candidate B:", rank_string(baseline_b))
print()
print("Without adjustment (0.19 - raw O*NET):")
print("  Candidate A:", rank_string(noadj_a))
print("  Candidate B:", rank_string(noadj_b))

# Find Content Creator's rank in each
cc_baseline_a_rank = [o for o, _ in baseline_a].index("Content Creator") + 1
cc_noadj_a_rank = [o for o, _ in noadj_a].index("Content Creator") + 1
cc_baseline_b_rank = [o for o, _ in baseline_b].index("Content Creator") + 1
cc_noadj_b_rank = [o for o, _ in noadj_b].index("Content Creator") + 1

print()
print(f"Content Creator rank for Candidate A: with={cc_baseline_a_rank}, without={cc_noadj_a_rank}")
print(f"Content Creator rank for Candidate B: with={cc_baseline_b_rank}, without={cc_noadj_b_rank}")
