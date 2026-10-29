"""bestofk: patient-clustered, selection-adjusted tests of "the best of K models beats a baseline"."""
from .audit import audit_claim, hanley_mcneil_se, rho_from_patient_level_ci, sensitivity_table
from .core import BestOfKResult, auc_influence, best_of_k_test, design_effect, threshold_influence

__version__ = "0.1.0"
__all__ = ["best_of_k_test", "BestOfKResult", "design_effect", "auc_influence", "threshold_influence", "audit_claim",
           "sensitivity_table", "hanley_mcneil_se", "rho_from_patient_level_ci", "__version__"]
