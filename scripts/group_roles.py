"""Group members and roles, as stated by the group (SPEC-project-completion.md, 2026-11-19).

One place for these facts. verify_readme.py, verify_guide.py and the report builder read them,
so the README, the viva guide and the report cannot disagree.
"""

MEMBERS = [
    # (name, roll number, role)
    ("Syed Muhammad Hunain", "CT-24281", "Model creation and documentation"),
    ("Muhammad Afzal", "CT-24263", "Frontend and deployments"),
    ("Ayesha", "CT-24309", "Backend"),
]

# Topic rows of the "Who answers what" table in docs/guides/demo_and_viva_guide.md: topic -> owner name.
# Each owner follows from the roles above; no authorship beyond those roles is claimed.
TOPIC_OWNERS = {
    "Data, EDA, aggregation (`features.py`, notebook)": "Syed Muhammad Hunain",
    "Pipeline, tuning, evaluation (`training.py`, `scripts/train.py`)": "Syed Muhammad Hunain",
    "Ranking and inference (`inference.py`)": "Syed Muhammad Hunain",
    "API and validation (`app/`)": "Ayesha",
    "Frontend pages (`frontend/`)": "Muhammad Afzal",
    "Deployments (Streamlit, Vercel, Hugging Face model repo; `streamlit_app/`, `hf_space/`, bundle scripts)": "Muhammad Afzal",
    "README, checks, report (`docs/`, `scripts/verify_*.py`)": "Syed Muhammad Hunain",
}

PLACEHOLDER = "(group to fill)"
