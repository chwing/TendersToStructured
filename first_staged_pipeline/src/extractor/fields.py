ALL_FIELDS = [
    # Identity
    "tender_id",
    "title",
    "reference_number",
    # Issuing Party
    "issuing_organization",
    "department",
    "country",
    "city_region",
    # Dates
    "publication_date",
    "submission_deadline",
    "questions_deadline",
    "award_date",
    # Financial
    "budget",
    "currency",
    "payment_terms",
    "financial_guarantee",
    # Technical Scope
    "project_description",
    "domain",
    "required_technologies",
    "deliverables",
    "scope_of_work",
    "hosting_requirements",
    # Team / HR Requirements
    "num_profiles",
    "roles_profiles",
    "seniority_level",
    "certifications",
    "mission_duration",
    # Contractor Eligibility
    "required_experience",
    "company_size",
    "required_documents",
    "geographic_restrictions",
    "legal_requirements",
    # Evaluation
    "evaluation_criteria",
    "lot_number",
    # Relevance
    "is_tech_relevant",
    "relevance_reason",
]

# Tier 1 — critical fields: full retrieval (top_k=8), temp=0, min_confidence=0.4
# These must never be lost. We trade context budget generously here.
MANDATORY_FIELDS = [
    "tender_id",
    "title",
    "reference_number",
    "issuing_organization",
    "publication_date",
    "submission_deadline",
    "budget",
    "project_description",
    "domain",
    "scope_of_work",
    "is_tech_relevant",
]

# Tier 2 — useful but recoverable: standard retrieval (top_k=5), temp=0, min_confidence=0.6
NICE_TO_HAVE_FIELDS = [
    "department",
    "country",
    "city_region",
    "questions_deadline",
    "award_date",
    "currency",
    "payment_terms",
    "required_technologies",
    "deliverables",
    "num_profiles",
    "roles_profiles",
    "mission_duration",
    "required_experience",
    "evaluation_criteria",
    "relevance_reason",
]

# Tier 3 — enrichment: light retrieval (top_k=3), temp=0.3 (allows inference), min_confidence=0.5
EXTRA_FIELDS = [
    "financial_guarantee",
    "hosting_requirements",
    "seniority_level",
    "certifications",
    "company_size",
    "required_documents",
    "geographic_restrictions",
    "legal_requirements",
    "lot_number",
]

FIELD_TIER: dict[str, str] = (
    {f: "mandatory" for f in MANDATORY_FIELDS}
    | {f: "nice_to_have" for f in NICE_TO_HAVE_FIELDS}
    | {f: "extra" for f in EXTRA_FIELDS}
)
