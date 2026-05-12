from dataclasses import dataclass


@dataclass
class PredictRequest:
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


@dataclass
class PredictResponse:
    loan_approved: bool
    approval_probability: float


@dataclass
class HealthResponse:
    status: str
    model_loaded: bool
