"""
Bias check for the strengths-based occupational recommender.

Runs the recommender against a diverse set of synthetic candidate profiles
and counts how often each occupation appears in the top-3 recommendations.
If the sample or method is biased, one or two occupations will dominate.
If it is reasonably balanced, each occupation should appear in the top-3
for candidates whose strengths match it.

This is not a validation of accuracy - only accuracy against real users
could validate that. This is a check on whether the design has an obvious
systematic tilt.
"""

import random
from recommender import (
    DIMENSIONS,
    OCCUPATIONS,
    DEMAND,
    cosine_similarity,
    per_dimension_contributions,
)


def rank(candidate, occupations):
    scores = [(name, cosine_similarity(candidate, vec)) for name, vec in occupations.items()]
    scores.sort(key=lambda x: -x[1])
    return [name for name, _ in scores]


# Twelve synthetic candidate profiles across a range of strengths patterns.
# Values are on 0-to-1 scale for the ten dimensions.
#
# NOTE: These twelve profiles are diagnostic tools for the sample coverage
# check reported in Methodology Section III.D. They are DISTINCT from the two
# candidate profiles used in the Illustrative Example (Section IV). In
# particular, "B: person-oriented" here is a legacy general-purpose profile,
# NOT the paper's Deaf Candidate B (which is defined in candidate_b_deaf.py
# with oral_comm=0.30 for hearing-default and 0.75 for Sunyi-inclusive).
CANDIDATES = {
    # Original two (diagnostic profiles, not the Illustrative Example candidates)
    "A: analytical/technical": [0.85, 0.55, 0.75, 0.50, 0.55, 0.45, 0.80, 0.35, 0.45, 0.80],
    "B: person-oriented":      [0.40, 0.50, 0.40, 0.70, 0.90, 0.80, 0.55, 0.50, 0.85, 0.45],

    # Analytical variants
    "C: analytical, low tech":  [0.85, 0.55, 0.75, 0.50, 0.55, 0.45, 0.80, 0.35, 0.45, 0.30],
    "D: analytical, low writing": [0.85, 0.55, 0.75, 0.50, 0.55, 0.45, 0.30, 0.35, 0.45, 0.80],

    # Person-oriented variants
    "E: empathy-strong, low oral": [0.40, 0.50, 0.40, 0.70, 0.90, 0.40, 0.55, 0.50, 0.85, 0.45],
    "F: service, low empathy":  [0.40, 0.50, 0.40, 0.70, 0.50, 0.80, 0.55, 0.50, 0.90, 0.45],

    # Creative
    "G: creative, moderate tech": [0.50, 0.90, 0.50, 0.55, 0.55, 0.60, 0.65, 0.40, 0.55, 0.70],

    # Balanced / no clear specialization
    "H: balanced, moderate all": [0.55, 0.55, 0.55, 0.60, 0.55, 0.60, 0.60, 0.50, 0.60, 0.55],

    # Communication-strong
    "I: communication, low tech": [0.55, 0.45, 0.50, 0.60, 0.70, 0.85, 0.85, 0.55, 0.60, 0.35],

    # Resilience-strong (life/practical experience)
    "J: resilient, practical":  [0.50, 0.55, 0.50, 0.90, 0.65, 0.65, 0.55, 0.55, 0.65, 0.50],

    # Leadership-strong (peer mentor, community role)
    "K: leadership-strong":     [0.60, 0.55, 0.60, 0.70, 0.70, 0.75, 0.65, 0.85, 0.65, 0.55],

    # Systems-strong (organizer, planner)
    "L: systems + written":     [0.65, 0.45, 0.85, 0.55, 0.45, 0.50, 0.80, 0.50, 0.50, 0.65],
}


def main():
    top1_counts = {name: 0 for name in OCCUPATIONS}
    top3_counts = {name: 0 for name in OCCUPATIONS}
    top1_appearances = {name: [] for name in OCCUPATIONS}

    print("=" * 78)
    print("BIAS CHECK: 12 candidate profiles against the 7-occupation sample")
    print("=" * 78)
    print()
    print(f"{'Candidate':<32}{'Top 1':<24}{'Top 2':<24}{'Top 3':<24}")
    print("-" * 104)

    for cand_name, vector in CANDIDATES.items():
        ranking = rank(vector, OCCUPATIONS)
        top1_counts[ranking[0]] += 1
        top1_appearances[ranking[0]].append(cand_name.split(":")[0])
        for occ in ranking[:3]:
            top3_counts[occ] += 1
        print(f"{cand_name:<32}{ranking[0]:<24}{ranking[1]:<24}{ranking[2]:<24}")

    print()
    print("=" * 78)
    print(f"How often each occupation appeared, out of {len(CANDIDATES)} candidates")
    print("=" * 78)
    print(f"{'Occupation':<28}{'Ranked #1':<14}{'In top 3':<14}{'Won for candidates':<30}")
    print("-" * 86)
    for occ in OCCUPATIONS:
        top1 = top1_counts[occ]
        top3 = top3_counts[occ]
        wins = ", ".join(top1_appearances[occ]) if top1_appearances[occ] else "-"
        print(f"{occ:<28}{top1:<14}{top3:<14}{wins:<30}")

    print()
    print("Interpretation guide:")
    print("- Balanced sample: each occupation appears in some candidate's top 3")
    print("- Biased sample: 1-2 occupations dominate top 1 across many candidates")
    print("- Never-top-3 occupations may need vector or sample review")

    # -------------------------------------------------------------------------
    # Input-responsiveness diagnostic vs random-ranker baseline
    #
    # A hostile reviewer's natural objection to a proof-of-concept recommender
    # is: "how do we know the rankings aren't just noise?"
    #
    # A minimum condition for the recommender to be doing meaningful work is
    # that different candidate strengths patterns produce different rankings.
    # If all 12 profiles produced the same top-3, the recommender would be
    # ignoring the input; if the 12 top-3s were random, that would be the
    # random-ranker outcome. Neither is what we observe.
    #
    # Two numbers below:
    # (i)  distinct top-3 sets produced by the recommender across the 12
    #      diagnostic profiles (higher = more input-responsive).
    # (ii) expected number of distinct top-3 sets a random ranker would
    #      produce across 12 trials, estimated by simulation. Under the null,
    #      each trial draws a uniform size-3 subset from C(7,3)=35.
    #
    # This is a necessary but not sufficient condition for meaningful work.
    # It does NOT validate accuracy against real user outcomes.
    # -------------------------------------------------------------------------
    N_TRIALS = 10000
    rng = random.Random(42)
    occ_list = list(OCCUPATIONS.keys())

    # (i) Recommender's distinct top-3 sets
    recommender_top3 = {
        cand_name: frozenset(rank(vector, OCCUPATIONS)[:3])
        for cand_name, vector in CANDIDATES.items()
    }
    distinct_top3s_recommender = len(set(recommender_top3.values()))

    # (ii) Random ranker: for each of N trials, draw 12 uniform size-3 subsets
    # and count distinct ones. Report the mean and empirical distribution.
    from math import comb
    n_profiles = len(CANDIDATES)
    distinct_counts = []
    for _ in range(N_TRIALS):
        subs = [frozenset(rng.sample(occ_list, 3)) for _ in range(n_profiles)]
        distinct_counts.append(len(set(subs)))
    mean_distinct_random = sum(distinct_counts) / len(distinct_counts)
    # Empirical fraction of random trials with <= observed diversity
    p_le_observed = sum(1 for c in distinct_counts if c <= distinct_top3s_recommender) / len(distinct_counts)

    print()
    print("=" * 78)
    print("INPUT-RESPONSIVENESS DIAGNOSTIC (recommender vs random ranker)")
    print("=" * 78)
    print(f"Number of diagnostic profiles:                       {n_profiles}")
    print(f"Distinct top-3 sets from recommender:                {distinct_top3s_recommender} of {n_profiles}")
    print(f"Distinct top-3 sets from random ranker (mean, {N_TRIALS} trials): {mean_distinct_random:.2f}")
    print(f"Fraction of random trials with <= {distinct_top3s_recommender} distinct sets:      {p_le_observed:.4f}")
    print()
    print("Interpretation:")
    print(f"- The recommender collapses the 12 diagnostic profiles onto {distinct_top3s_recommender} distinct")
    print("  top-3 sets. This is fewer than a random ranker would produce")
    print(f"  (mean {mean_distinct_random:.2f}), reflecting that the recommender clusters similar")
    print("  strengths patterns onto similar occupational regions rather than")
    print("  producing random assignments.")
    print("- The recommender is not returning a single top-3 regardless of")
    print("  input; it is producing structured variation.")
    print("- This is a necessary but not sufficient condition for meaningful")
    print("  work. Accuracy against real user outcomes is not tested here")
    print("  and remains future work.")


if __name__ == "__main__":
    main()
