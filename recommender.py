"""
Strengths-based occupational recommender: proof of concept.

Runs cosine similarity between a candidate strengths vector and a set of
occupation skill vectors, and prints ranked recommendations with
per-dimension explanations. Reports demand as an independent signal
alongside similarity, not combined into a single score.
"""

import math

# Ten prototype skill dimensions from Table I
DIMENSIONS = [
    "Analytical thinking",
    "Creative thinking",
    "Systems thinking",
    "Resilience / flexibility",
    "Empathy & active listening",
    "Communication (oral)",
    "Communication (written)",
    "Leadership & social influence",
    "Service orientation",
    "Technological literacy",
]

# Occupation vectors anchored in O*NET Skills Importance ratings (rescaled
# 0-100 -> 0-1). Each dimension is the max of its mapped O*NET skills.
# Mapping documented in onet_mapping.md; computation in compute_vectors.py.
# Resilience assigned from BLS narrative (O*NET has no direct Skills entry).
# One documented adjustment: Content Creator Creative raised from 0.19 to 0.75
# to reflect Indonesian entry-level content creator context.
OCCUPATIONS = {
    "Customer Service Rep":        [0.56, 0.22, 0.31, 0.70, 0.75, 0.72, 0.60, 0.50, 0.75, 0.22],
    "Retail Sales Assistant":      [0.53, 0.22, 0.25, 0.65, 0.69, 0.69, 0.50, 0.72, 0.69, 0.25],
    "Administrative Assistant":    [0.50, 0.22, 0.25, 0.55, 0.75, 0.75, 0.72, 0.41, 0.63, 0.22],
    "Content Creator":             [0.66, 0.75, 0.35, 0.60, 0.72, 0.72, 0.94, 0.60, 0.53, 0.19],
    "IT Support Specialist":       [0.69, 0.35, 0.50, 0.65, 0.75, 0.75, 0.75, 0.50, 0.53, 0.50],
    "Food Service Worker":         [0.47, 0.00, 0.25, 0.75, 0.63, 0.63, 0.44, 0.44, 0.72, 0.25],
    "Software Developer":          [0.72, 0.60, 0.63, 0.55, 0.60, 0.53, 0.63, 0.47, 0.47, 0.75],
}

# BLS 2024-2034 Employment Projections. Growth is percent change in
# employment 2024-2034; openings is annual average occupational openings
# 2024-2034. All figures are for the specific SOC code identified for each
# occupation. Source: BLS Table 1.2 Occupational projections, 2024-2034
# (https://www.bls.gov/emp/tables/occupational-projections-and-characteristics.htm),
# last updated 28 August 2025. Growth percentages are rounded to the
# nearest whole percent for reporting.
DEMAND = {
    # 43-4051 Customer Service Representatives: -5.5% -> -5%, 341,700
    "Customer Service Rep":      {"growth_pct": -5.0,  "annual_openings": 341700},
    # 41-2031 Retail Salespersons: -0.5% -> -1%, 555,800
    "Retail Sales Assistant":    {"growth_pct": -1.0,  "annual_openings": 555800},
    # 43-6014 Secretaries and Administrative Assistants, Except Legal,
    # Medical, and Executive: -1.6% -> -2%, 202,800
    "Administrative Assistant":  {"growth_pct": -2.0,  "annual_openings": 202800},
    # 27-3043 Writers and Authors: 3.6% -> +4%, 13,400
    "Content Creator":           {"growth_pct":  4.0,  "annual_openings": 13400},
    # 15-1232 Computer User Support Specialists: -3.7% -> -4%, 40,800
    "IT Support Specialist":     {"growth_pct": -4.0,  "annual_openings": 40800},
    # 35-3031 Waiters and Waitresses: -0.7% -> -1%, 456,700
    "Food Service Worker":       {"growth_pct": -1.0,  "annual_openings": 456700},
    # 15-1252 Software Developers: 15.8% -> +16%, 115,200
    "Software Developer":        {"growth_pct": 16.0,  "annual_openings": 115200},
}

# Candidate A (synthetic, illustrative) - analytical/technical strengths profile
CANDIDATE_A = [0.85, 0.55, 0.75, 0.50, 0.55, 0.45, 0.80, 0.35, 0.45, 0.80]

# Candidate B (synthetic, illustrative) - legacy person-oriented strengths
# profile. Retained for backward compatibility with earlier iterations of the
# recommender. THE PAPER'S ILLUSTRATIVE EXAMPLE USES THE DEAF CANDIDATE B
# DEFINED IN candidate_b_deaf.py (CANDIDATE_B_DEAF and CANDIDATE_B_SUNYI),
# not this vector.
CANDIDATE_B_LEGACY = [0.40, 0.50, 0.40, 0.70, 0.90, 0.80, 0.55, 0.50, 0.85, 0.45]


def l2_normalize(v):
    """Rescale a vector so it has length 1. This lets cosine similarity focus
    on the shape of the strengths profile rather than its overall magnitude."""
    magnitude = math.sqrt(sum(x * x for x in v))
    if magnitude == 0:
        return v
    return [x / magnitude for x in v]


def cosine_similarity(a, b):
    """Cosine of the angle between two vectors. Returns a value in [0, 1] for
    non-negative vectors. Higher means the strengths profiles have their high
    values in similar places."""
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    return sum(x * y for x, y in zip(a_norm, b_norm))


def per_dimension_contributions(candidate, occupation):
    """Break the cosine similarity down by dimension so we can explain which
    dimensions drove the match."""
    c_norm = l2_normalize(candidate)
    o_norm = l2_normalize(occupation)
    contributions = []
    for i, dim in enumerate(DIMENSIONS):
        contributions.append((dim, c_norm[i] * o_norm[i]))
    return sorted(contributions, key=lambda x: -x[1])


def min_max_scale(values):
    """Rescale a list of numbers to sit between 0 and 1 across the sample."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def demand_score(demand_data):
    """Combine growth and openings into a single demand score with equal
    weighting. Used as a secondary contextual signal, not for ranking."""
    occ_names = list(demand_data.keys())
    growths = [demand_data[o]["growth_pct"] for o in occ_names]
    openings = [demand_data[o]["annual_openings"] for o in occ_names]
    g_scaled = min_max_scale(growths)
    a_scaled = min_max_scale(openings)
    return {occ_names[i]: 0.5 * g_scaled[i] + 0.5 * a_scaled[i] for i in range(len(occ_names))}


def recommend(candidate, occupations, demand_data, top_k=None, show_contribs=3):
    """Produce ranked recommendations for a candidate."""
    scores = []
    for name, vec in occupations.items():
        score = cosine_similarity(candidate, vec)
        scores.append((name, score, vec))
    scores.sort(key=lambda x: -x[1])

    demand = demand_score(demand_data)

    print("=" * 78)
    print(f"{'Rank':<6}{'Occupation':<28}{'Skill match':<14}{'Demand':<10}")
    print("-" * 78)
    for rank, (name, score, vec) in enumerate(scores[:top_k] if top_k else scores, 1):
        d = demand[name]
        print(f"{rank:<6}{name:<28}{score:<14.3f}{d:<10.3f}")
    print("=" * 78)
    print()

    # Explanations for the top 3
    print("Why each of the top three matches:")
    print("-" * 78)
    for rank, (name, score, vec) in enumerate(scores[:3], 1):
        contribs = per_dimension_contributions(candidate, vec)
        top_dims = contribs[:show_contribs]
        print(f"\n{rank}. {name} (skill match = {score:.3f})")
        print(f"   Top contributing dimensions:")
        for dim, contrib in top_dims:
            print(f"     - {dim}: {contrib:.3f}")

    print()
    print("Demand context (BLS 2024-2034 projections):")
    print("-" * 78)
    for name, _, _ in scores:
        d = demand_data[name]
        print(f"  {name}: {d['growth_pct']:+.0f}% growth, ~{d['annual_openings']:,} annual openings")


def show_candidate(name, description, vector):
    print(f"\n{name}: {description}")
    print("Strengths profile:")
    for dim, val in zip(DIMENSIONS, vector):
        bar = "#" * int(val * 20)
        print(f"  {dim:<32} {val:.2f}  {bar}")
    print()


if __name__ == "__main__":
    print("\nStrengths-based occupational recommender: proof of concept")
    print("=" * 78)

    show_candidate(
        "Candidate A",
        "final-year undergraduate, analytical/technical strengths",
        CANDIDATE_A,
    )
    recommend(CANDIDATE_A, OCCUPATIONS, DEMAND)

    print("\n\n")

    # The paper's Candidate B (Deaf SMK graduate, hospitality track) with both
    # scorings is defined and run in candidate_b_deaf.py. Run that script to
    # reproduce the Candidate B rankings reported in Section IV.
    #
    # The legacy person-oriented profile below is retained only for backward
    # compatibility with earlier iterations. It is NOT the paper's Candidate B
    # and its scores do not match the ones reported in Section IV.
    print("\nFor the paper's Candidate B (Deaf SMK graduate, hearing-default")
    print("and Sunyi-inclusive scorings), run: python candidate_b_deaf.py\n")

    show_candidate(
        "Legacy person-oriented profile (NOT the paper's Candidate B)",
        "retained for backward compatibility; see candidate_b_deaf.py for the paper's B",
        CANDIDATE_B_LEGACY,
    )
    recommend(CANDIDATE_B_LEGACY, OCCUPATIONS, DEMAND)
