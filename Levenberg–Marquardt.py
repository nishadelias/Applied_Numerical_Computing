import numpy as np
from scipy.optimize import least_squares

# Data points
points = np.array([
    [0.5, 1.5], [-0.3, 0.6], [1.0, 1.8], [-0.4, 0.2], [0.2, 1.3],
    [0.7, 0.1], [2.3, 0.8], [1.4, 0.5], [0.0, 0.2], [2.4, 1.7]
])

# Initial guess for ellipse
c1, c2 = np.mean(points, axis=0)
r = 1.0
δ = 0.0
alpha = 0.0

# Initial guess for t 
t_init = np.arctan2(points[:, 1] - c2, points[:, 0] - c1)

# Parameter vector:
params_init = np.hstack([c1, c2, r, δ, alpha, t_init])

# Ellipse func
def f(t, c1, c2, r, δ, alpha):
    x = c1 + r * np.cos(alpha + t) + δ * np.cos(alpha - t)
    y = c2 + r * np.sin(alpha + t) + δ * np.sin(alpha - t)
    return np.vstack([x, y]).T

# Residual func
def residuals(params, points):
    c1, c2, r, δ, alpha = params[:5]
    t = params[5:]
    ellipse_points = f(t, c1, c2, r, δ, alpha)
    return (ellipse_points - points).ravel()

# Levenberg-Marquardt to minimize the residuals
result = least_squares(residuals, params_init, method='lm', args=(points,))

# Optimized parameters
c1_opt, c2_opt, r_opt, δ_opt, alpha_opt = result.x[:5]
t_opt = result.x[5:]

print("Optimized center:", (c1_opt, c2_opt))
print("Optimized radius:", r_opt)
print("Optimized δ:", δ_opt)
print("Optimized alpha:", alpha_opt)
