from typing import Any, Optional
from pydantic import BaseModel, field_validator


class ExtractedField(BaseModel):
    value: Any
    confidence: float  # 0.90=high, 0.65=medium, 0.40=low

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v):
        if isinstance(v, str):
            mapping = {"high": 0.90, "medium": 0.65, "low": 0.40}
            return mapping.get(v.lower(), 0.40)
        return v


class TenderExtraction(BaseModel):
    source_file: str
    document_language: Optional[str] = None

    # Identity
    tender_id: Optional[ExtractedField] = None
    title: Optional[ExtractedField] = None
    reference_number: Optional[ExtractedField] = None

    # Issuing Party
    issuing_organization: Optional[ExtractedField] = None
    department: Optional[ExtractedField] = None
    country: Optional[ExtractedField] = None
    city_region: Optional[ExtractedField] = None

    # Dates
    publication_date: Optional[ExtractedField] = None
    submission_deadline: Optional[ExtractedField] = None
    questions_deadline: Optional[ExtractedField] = None
    award_date: Optional[ExtractedField] = None

    # Financial
    budget: Optional[ExtractedField] = None
    currency: Optional[ExtractedField] = None
    payment_terms: Optional[ExtractedField] = None
    financial_guarantee: Optional[ExtractedField] = None

    # Technical Scope
    project_description: Optional[ExtractedField] = None
    domain: Optional[ExtractedField] = None
    required_technologies: Optional[ExtractedField] = None
    deliverables: Optional[ExtractedField] = None
    scope_of_work: Optional[ExtractedField] = None
    hosting_requirements: Optional[ExtractedField] = None

    # Team / HR Requirements
    num_profiles: Optional[ExtractedField] = None
    roles_profiles: Optional[ExtractedField] = None
    seniority_level: Optional[ExtractedField] = None
    certifications: Optional[ExtractedField] = None
    mission_duration: Optional[ExtractedField] = None

    # Contractor Eligibility
    required_experience: Optional[ExtractedField] = None
    company_size: Optional[ExtractedField] = None
    required_documents: Optional[ExtractedField] = None
    geographic_restrictions: Optional[ExtractedField] = None
    legal_requirements: Optional[ExtractedField] = None

    # Evaluation
    evaluation_criteria: Optional[ExtractedField] = None
    lot_number: Optional[ExtractedField] = None

    # Relevance
    is_tech_relevant: Optional[ExtractedField] = None
    relevance_reason: Optional[ExtractedField] = None
