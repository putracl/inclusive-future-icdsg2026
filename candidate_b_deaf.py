"""
Run recommender on Candidate B redefined as an explicit Deaf SMK student
in a hospitality track, trained in a Sunyi Coffee-style inclusive workplace model.

Strengths scored honestly on the ten dimensions as designed:
- Oral communication is spoken-language framed by O*NET's Speaking skill.
  For a Deaf candidate whose primary language is BISINDO, this dimension
  is scored low (0.30) - not because they lack communication ability,
  but because the DIMENSION as constructed measures spoken language.
- Empathy and active listening scored high (0.85) - visual attention,
  community-oriented listening, facial-cue reading are real strengths.
- The gap between "cannot speak Indonesian fluently" and "has strong
  interpersonal communication capacity" is exactly what the recommender's
  current dimensions do not capture. That gap is the point.
"""

import math
from recommender import OCCUPATIONS, DIMENSIONS

# Candidate B redefined - Deaf SMK graduate, hospitality/service track,
# trained in Sunyi Coffee-style inclusive workplace
CANDIDATE_B_DEAF = [
    0.45,  # Analytical thinking
    0.55,  # Creative thinking
    0.40,  # Systems thinking
    0.85,  # Resilience/flexibility - high, real
    0.85,  # Empathy & active listening - high (visual, community-oriented)
    0.30,  # Communication (oral) - low: spoken Indonesian limited; sign primary
    0.55,  # Communication (written) - moderate
    0.45,  # Leadership & social influence
    0.85,  # Service orientation - high
    0.45,  # Technological literacy
]


def l2(v):
    m = math.sqrt(sum(x * x for x in v))
    return v if m == 0 else [x / m for x in v]


def cosine(a, b):
    an, bn = l2(a), l2(b)
    return sum(x * y for x, y in zip(an, bn))


def per_dim_contributions(cand, occ_vec):
    """Cosine similarity broken down per dimension."""
    cn = l2(cand)
    on = l2(occ_vec)
    return [c * o for c, o in zip(cn, on)]


# Run the ranking
print("=" * 78)
print("CANDIDATE B (Deaf) - strengths on the ten dimensions")
print("=" * 78)
for dim, val in zip(DIMENSIONS, CANDIDATE_B_DEAF):
    print(f"  {dim:<40} {val:.2f}")

print()
print("=" * 78)
print("RANKING")
print("=" * 78)
scored = [(occ, cosine(CANDIDATE_B_DEAF, v)) for occ, v in OCCUPATIONS.items()]
scored.sort(key=lambda x: -x[1])

for i, (occ, score) in enumerate(scored, 1):
    print(f"  {i}. {occ:<28} {score:.3f}")

print()
print("=" * 78)
print("PER-DIMENSION CONTRIBUTIONS FOR TOP 3")
print("=" * 78)
for occ, score in scored[:3]:
    contribs = per_dim_contributions(CANDIDATE_B_DEAF, OCCUPATIONS[occ])
    print(f"\n{occ} (total {score:.3f}):")
    ranked_dims = sorted(zip(DIMENSIONS, contribs), key=lambda x: -x[1])
    for dim, c in ranked_dims:
        print(f"  {dim:<40} {c:.3f}")


# --- Compare to a Sunyi-inclusive workplace scenario ---
# In a workplace where oral communication is done in BISINDO, the "oral
# communication" dimension effectively rescales - the candidate DOES have
# oral communication capacity, just in a different modality. Show what
# happens if we score oral communication as 0.75 to represent that scenario.
print()
print("=" * 78)
print("SUNYI-INCLUSIVE SCENARIO: oral communication rescaled")
print("=" * 78)
print("If the workplace communicates in BISINDO, oral comm effectively = 0.75")
print()
CANDIDATE_B_SUNYI = CANDIDATE_B_DEAF.copy()
CANDIDATE_B_SUNYI[5] = 0.75  # oral communication in signed workplace

scored2 = [(occ, cosine(CANDIDATE_B_SUNYI, v)) for occ, v in OCCUPATIONS.items()]
scored2.sort(key=lambda x: -x[1])
for i, (occ, score) in enumerate(scored2, 1):
    delta_score = score - dict(scored)[occ]
    print(f"  {i}. {occ:<28} {score:.3f}  ({delta_score:+.3f})")

print()
print("Rank movement (default -> Sunyi-inclusive):")
default_ranks = {o: i for i, (o, _) in enumerate(scored, 1)}
sunyi_ranks = {o: i for i, (o, _) in enumerate(scored2, 1)}
for occ in OCCUPATIONS:
    d = default_ranks[occ]
    s = sunyi_ranks[occ]
    if d != s:
        print(f"  {occ}: rank {d} -> {s}")
    else:
        print(f"  {occ}: rank {d} (unchanged)")
