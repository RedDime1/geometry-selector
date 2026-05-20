import numpy as np

from geomselect import fit_spherical
from tests.test_selector import make_sphere_test_data

D, U_true, R_true = make_sphere_test_data(
    n=80,
    d=2,
    R=3.0,
    seed=2,
    noise=0.0,
)

candidate = fit_spherical(
    D,
    d=2,
    grid_num=31,
    n_refine=3,
    refine_num=25,
)

selection = candidate.metadata["selection"]
scale_s = candidate.metadata["scale_s"]

R_selected = candidate.parameter_value
R_scaled_selected = candidate.metadata["R_scaled"]
R_scaled_true = R_true / scale_s

print("R_true:", R_true)
print("R_selected:", R_selected)
print("R_scaled_true:", R_scaled_true)
print("R_scaled_selected:", R_scaled_selected)
print("stress_selected:", candidate.stress)

D_scaled = D / scale_s

U_at_true, _, _, _ = spherical_fixed_R_eigsh(
    D_scaled,
    d=2,
    R=R_scaled_true,
)

stress_at_true = spherical_stress(
    D,
    U_at_true,
    R=R_true,
)

print("stress_at_true_R:", stress_at_true)

print("score_selected:", spherical_min_score(D_scaled, R_scaled_selected))
print("score_true:", spherical_min_score(D_scaled, R_scaled_true))
print("lambda_min_selected:", _safe_min_eigval(np.cos(D_scaled / R_scaled_selected)))
print("lambda_min_true:", _safe_min_eigval(np.cos(D_scaled / R_scaled_true)))