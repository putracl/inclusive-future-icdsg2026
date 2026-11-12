"""
Compute occupation vectors from O*NET Skills Importance ratings.

Uses the mapping documented in onet_mapping.md:
- For each dimension, take the MAX of the mapped O*NET skills' Importance values
- Rescale from 0-100 to 0-1
- Apply documented adjustments where noted

Prints both the vector and the intermediate O*NET values for verification.
"""

# O*NET Skills Importance data (0-100 scale) for each occupation
# Source: onetonline.org, fetched Aug 2026
ONET_DATA = {
    "Customer Service Rep": {  # 43-4051.00
        "Active Listening": 75, "Speaking": 72, "Reading Comprehension": 60,
        "Writing": 50, "Critical Thinking": 56, "Active Learning": 47,
        "Monitoring": 50, "Learning Strategies": 41, "Mathematics": 31, "Science": 0,
        "Service Orientation": 75, "Complex Problem Solving": 53, "Time Management": 53,
        "Negotiation": 50, "Persuasion": 50, "Social Perceptiveness": 50,
        "Coordination": 47, "Judgment and Decision Making": 47, "Instructing": 44,
        "Management of Personnel Resources": 35, "Systems Analysis": 31,
        "Systems Evaluation": 31, "Technology Design": 22,
        "Operations Analysis": 22, "Operations Monitoring": 0, "Troubleshooting": 0,
        "Programming": 0, "Quality Control Analysis": 0,
    },
    "Retail Sales Assistant": {  # 41-2031.00
        "Active Listening": 69, "Speaking": 69, "Reading Comprehension": 50,
        "Writing": 50, "Critical Thinking": 53, "Active Learning": 50,
        "Monitoring": 50, "Learning Strategies": 41, "Mathematics": 41, "Science": 0,
        "Persuasion": 72, "Service Orientation": 69, "Negotiation": 63,
        "Social Perceptiveness": 63, "Coordination": 50, "Time Management": 50,
        "Instructing": 47, "Judgment and Decision Making": 47,
        "Complex Problem Solving": 44, "Management of Personnel Resources": 35,
        "Systems Analysis": 25, "Systems Evaluation": 25, "Technology Design": 22,
        "Troubleshooting": 19, "Operations Analysis": 13, "Operations Monitoring": 25,
        "Programming": 6, "Quality Control Analysis": 25,
    },
    "Administrative Assistant": {  # 43-6014.00
        "Active Listening": 75, "Speaking": 75, "Reading Comprehension": 72,
        "Writing": 69, "Critical Thinking": 50, "Active Learning": 47,
        "Monitoring": 53, "Learning Strategies": 28, "Mathematics": 25, "Science": 0,
        "Service Orientation": 63, "Time Management": 63, "Coordination": 53,
        "Judgment and Decision Making": 50, "Social Perceptiveness": 50,
        "Complex Problem Solving": 47, "Management of Personnel Resources": 41,
        "Negotiation": 35, "Persuasion": 31, "Instructing": 28,
        "Systems Analysis": 25, "Systems Evaluation": 25,
        "Operations Analysis": 22, "Operations Monitoring": 22,
        "Quality Control Analysis": 22, "Programming": 16, "Technology Design": 13,
        "Troubleshooting": 0,
    },
    "Content Creator": {  # 27-3043.00 Writers and Authors
        "Writing": 94, "Reading Comprehension": 78, "Active Listening": 72,
        "Speaking": 72, "Critical Thinking": 66, "Active Learning": 60,
        "Monitoring": 50, "Learning Strategies": 44, "Mathematics": 6, "Science": 3,
        "Time Management": 66, "Coordination": 60, "Persuasion": 60,
        "Social Perceptiveness": 60, "Judgment and Decision Making": 56,
        "Service Orientation": 53, "Complex Problem Solving": 50,
        "Instructing": 47, "Negotiation": 47,
        "Management of Personnel Resources": 38, "Systems Evaluation": 35,
        "Systems Analysis": 31, "Technology Design": 19, "Programming": 16,
        "Quality Control Analysis": 13, "Operations Analysis": 6,
        "Operations Monitoring": 0, "Troubleshooting": 0,
    },
    "IT Support Specialist": {  # 15-1232.00
        "Active Listening": 75, "Reading Comprehension": 75, "Speaking": 75,
        "Critical Thinking": 69, "Writing": 63, "Active Learning": 53,
        "Learning Strategies": 53, "Monitoring": 53, "Mathematics": 31, "Science": 19,
        "Complex Problem Solving": 66, "Judgment and Decision Making": 56,
        "Service Orientation": 53, "Social Perceptiveness": 53, "Instructing": 50,
        "Operations Monitoring": 50, "Systems Analysis": 50, "Time Management": 50,
        "Troubleshooting": 50, "Coordination": 47, "Quality Control Analysis": 47,
        "Systems Evaluation": 47, "Persuasion": 41,
        "Management of Personnel Resources": 35, "Negotiation": 35,
        "Operations Analysis": 35, "Programming": 31, "Technology Design": 25,
    },
    "Food Service Worker": {  # 35-3031.00 Waiters and Waitresses
        "Service Orientation": 72, "Active Listening": 63, "Speaking": 63,
        "Social Perceptiveness": 60, "Coordination": 53, "Monitoring": 50,
        "Critical Thinking": 47, "Judgment and Decision Making": 44, "Persuasion": 44,
        "Reading Comprehension": 44, "Time Management": 44,
        "Complex Problem Solving": 41, "Active Learning": 38,
        "Quality Control Analysis": 31, "Instructing": 28, "Mathematics": 28,
        "Negotiation": 28, "Writing": 28, "Learning Strategies": 25,
        "Operations Monitoring": 25, "Systems Analysis": 25, "Systems Evaluation": 25,
        "Management of Personnel Resources": 22, "Troubleshooting": 19,
        "Programming": 3, "Operations Analysis": 0, "Science": 0, "Technology Design": 0,
    },
    "Software Developer": {  # 15-1252.00
        "Critical Thinking": 72, "Active Learning": 63, "Reading Comprehension": 63,
        "Active Listening": 60, "Writing": 56, "Speaking": 53, "Monitoring": 50,
        "Mathematics": 44, "Learning Strategies": 41, "Science": 28,
        "Programming": 75, "Judgment and Decision Making": 66,
        "Systems Analysis": 63, "Complex Problem Solving": 60, "Systems Evaluation": 60,
        "Technology Design": 60, "Operations Analysis": 56, "Time Management": 53,
        "Coordination": 50, "Instructing": 47, "Service Orientation": 47,
        "Social Perceptiveness": 47, "Operations Monitoring": 44, "Persuasion": 44,
        "Quality Control Analysis": 44, "Negotiation": 41, "Troubleshooting": 41,
        "Management of Personnel Resources": 31,
    },
}

# Mapping from ten dimensions to O*NET skills (max aggregation)
MAPPING = {
    "Analytical": ["Critical Thinking", "Complex Problem Solving",
                   "Judgment and Decision Making", "Mathematics"],
    "Creative": ["Operations Analysis", "Technology Design"],
    "Systems": ["Systems Analysis", "Systems Evaluation"],
    "Empathy": ["Active Listening", "Social Perceptiveness"],
    "OralComm": ["Speaking"],
    "WrittenComm": ["Writing", "Reading Comprehension"],
    "Leadership": ["Persuasion", "Negotiation",
                   "Management of Personnel Resources", "Instructing"],
    "Service": ["Service Orientation"],
    "TechLit": ["Programming", "Technology Design", "Troubleshooting",
                "Operations Monitoring"],
}

# Resilience values assigned from BLS narrative (see onet_mapping.md)
RESILIENCE = {
    "Customer Service Rep": 0.70,
    "Retail Sales Assistant": 0.65,
    "Administrative Assistant": 0.55,
    "Content Creator": 0.60,
    "IT Support Specialist": 0.65,
    "Food Service Worker": 0.75,
    "Software Developer": 0.55,
}

# Documented adjustments applied after O*NET mapping
ADJUSTMENTS = {
    "Content Creator": {
        "Creative": (0.75, "Indonesian entry-level content creator context "
                            "emphasizes creativity beyond what BLS Writers "
                            "and Authors captures"),
    },
}


def compute_vector(occupation_name):
    """Compute the ten-dimension vector for one occupation."""
    onet = ONET_DATA[occupation_name]
    vector = {}
    intermediate = {}

    for dim, skills in MAPPING.items():
        values = [onet.get(skill, 0) for skill in skills]
        max_value = max(values) if values else 0
        vector[dim] = round(max_value / 100.0, 2)
        intermediate[dim] = {"skills_used": {s: onet.get(s, 0) for s in skills},
                             "max": max_value, "rescaled": vector[dim]}

    # Insert resilience from separate table
    vector["Resilience"] = RESILIENCE[occupation_name]
    intermediate["Resilience"] = {"source": "BLS narrative, researcher judgment",
                                   "value": vector["Resilience"]}

    # Apply adjustments
    if occupation_name in ADJUSTMENTS:
        for dim, (new_val, reason) in ADJUSTMENTS[occupation_name].items():
            intermediate[dim]["adjusted_from"] = vector[dim]
            intermediate[dim]["adjusted_to"] = new_val
            intermediate[dim]["reason"] = reason
            vector[dim] = new_val

    return vector, intermediate


# Order: matches recommender.py dimension order
ORDER = ["Analytical", "Creative", "Systems", "Resilience", "Empathy",
         "OralComm", "WrittenComm", "Leadership", "Service", "TechLit"]

OCCUPATION_ORDER = [
    "Customer Service Rep",
    "Retail Sales Assistant",
    "Administrative Assistant",
    "Content Creator",
    "IT Support Specialist",
    "Food Service Worker",
    "Software Developer",
]


def main():
    print("=" * 80)
    print("O*NET-anchored occupation vectors")
    print("=" * 80)
    print()

    header = ["Occupation"] + ORDER
    widths = [24, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    row = "".join(h.ljust(w) for h, w in zip(header, widths))
    print(row)
    print("-" * sum(widths))

    print()
    print("Recommender-format vectors (Python literal):")
    print("OCCUPATIONS = {")

    for occ in OCCUPATION_ORDER:
        vec, _ = compute_vector(occ)
        values = [f"{vec[dim]:.2f}" for dim in ORDER]
        formatted = ", ".join(values)
        # Match the padding style of recommender.py
        name_field = f'    "{occ}":'.ljust(35)
        print(f'{name_field}[{formatted}],')

    print("}")
    print()

    # Also print the per-occupation intermediate for verification
    print("=" * 80)
    print("Intermediate values (for verification)")
    print("=" * 80)
    for occ in OCCUPATION_ORDER:
        vec, intermediate = compute_vector(occ)
        print()
        print(f"### {occ}")
        for dim in ORDER:
            info = intermediate[dim]
            if "skills_used" in info:
                skills_str = ", ".join(f"{s}={v}" for s, v in info["skills_used"].items())
                line = f"  {dim}: max({skills_str}) = {info['max']} -> {info['rescaled']}"
                if "adjusted_to" in info:
                    line += f" [ADJUSTED to {info['adjusted_to']}]"
                print(line)
            else:
                print(f"  {dim}: {info}")


if __name__ == "__main__":
    main()
