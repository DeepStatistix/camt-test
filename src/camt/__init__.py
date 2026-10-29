"""camt: the clustered AUC maximum test (CAMT), a patient-clustered, selection-adjusted test of "the best of K models beats a baseline"."""
from .audit import audit_claim, hanley_mcneil_se, rho_from_patient_level_ci, sensitivity_table
from .core import CamtResult, auc_influence, camt_test, design_effect, threshold_influence

__version__ = "0.1.0"
__all__ = ["camt_test", "CamtResult", "design_effect", "auc_influence", "threshold_influence", "audit_claim",
           "sensitivity_table", "hanley_mcneil_se", "rho_from_patient_level_ci", "__version__"]
