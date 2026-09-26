import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

# -------------------------------------------------------------
# Left Panel: Softmax Failure (Unbounded Polyhedral Cones)
# -------------------------------------------------------------
# Decision boundary lines through origin dividing into open cones
x_range = np.linspace(-4, 4, 200)
ax1.plot([-4, 4], [-2, 2], 'k--', lw=1.5, alpha=0.7)
ax1.plot([-4, 4], [3, -3], 'k--', lw=1.5, alpha=0.7)
ax1.plot([0, 0], [-4, 4], 'k--', lw=1.5, alpha=0.7)

# Class prototypes/clusters
c1 = np.array([2.0, 0.5])
c2 = np.array([-1.5, 2.0])
c3 = np.array([-1.5, -2.0])

ax1.scatter(c1[0] + np.random.normal(0, 0.3, 25), c1[1] + np.random.normal(0, 0.3, 25), color='#2b6cb0', alpha=0.6, s=30, label='Task 1 In-Dist.')
ax1.scatter(c2[0] + np.random.normal(0, 0.3, 25), c2[1] + np.random.normal(0, 0.3, 25), color='#c53030', alpha=0.6, s=30, label='Task 2 In-Dist.')
ax1.scatter(c3[0] + np.random.normal(0, 0.3, 25), c3[1] + np.random.normal(0, 0.3, 25), color='#2f855a', alpha=0.6, s=30, label='Task 3 In-Dist.')

# OOD point in infinite tail of cone
ood_point = np.array([3.6, 2.5])
ax1.scatter(ood_point[0], ood_point[1], color='#d69e2e', marker='*', s=250, edgecolor='black', zorder=5, label='OOD Test Input $x$')

# Annotations
ax1.annotate('Infinite Open Polyhedral Cone\nLinear Hyperplane: $W^T x + b$\n$P(y=1|x) \\approx 0.99$ (OOD Misrouting)', 
             xy=(ood_point[0], ood_point[1]), xytext=(0.1, 3.2),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
             fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec="#feb2b2"))

ax1.set_xlim(-4, 4.5)
ax1.set_ylim(-4, 4.5)
ax1.set_title("(a) Unbounded Polyhedral Cones (Softmax)\nLinear Hyperplanes $\\rightarrow$ OOD Extrapolation Failure", fontsize=11, fontweight='bold', color='#742a2a')
ax1.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.5)

# -------------------------------------------------------------
# Right Panel: SMR Conformal ETF Routing (Compact Ellipsoidal Voronoi Cells)
# -------------------------------------------------------------
from matplotlib.patches import Ellipse

# Plot closed compact ellipses around prototypes
e1 = Ellipse(c1, width=1.8, height=1.2, angle=20, edgecolor='#2b6cb0', facecolor='#ebf8ff', lw=2, alpha=0.7, label=r'Task 1 Voronoi Cell ($d_M^2 \leq \hat{q}$)')
e2 = Ellipse(c2, width=1.7, height=1.3, angle=-30, edgecolor='#c53030', facecolor='#fff5f5', lw=2, alpha=0.7, label=r'Task 2 Voronoi Cell ($d_M^2 \leq \hat{q}$)')
e3 = Ellipse(c3, width=1.9, height=1.1, angle=15, edgecolor='#2f855a', facecolor='#f0fff4', lw=2, alpha=0.7, label=r'Task 3 Voronoi Cell ($d_M^2 \leq \hat{q}$)')

ax2.add_patch(e1)
ax2.add_patch(e2)
ax2.add_patch(e3)

# Scatter points inside
ax2.scatter(c1[0] + np.random.normal(0, 0.2, 25), c1[1] + np.random.normal(0, 0.2, 25), color='#2b6cb0', alpha=0.7, s=30)
ax2.scatter(c2[0] + np.random.normal(0, 0.2, 25), c2[1] + np.random.normal(0, 0.2, 25), color='#c53030', alpha=0.7, s=30)
ax2.scatter(c3[0] + np.random.normal(0, 0.2, 25), c3[1] + np.random.normal(0, 0.2, 25), color='#2f855a', alpha=0.7, s=30)

# OOD point in rejection space
ax2.scatter(ood_point[0], ood_point[1], color='#d69e2e', marker='*', s=250, edgecolor='black', zorder=5, label='OOD Input x (Rejected)')

ax2.annotate('Low-Density Rejection Space\n$d_M^2(x, \\mu_c) > \\hat{q}_{\\alpha}$\nBounded Ellipsoid: $|\\mathcal{C}(x)| \\leq 2$\nGuaranteed Coverage: $\\geq 95.0\\%$', 
             xy=(ood_point[0], ood_point[1]), xytext=(0.2, 2.5),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
             fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#f0fff4", ec="#9ae6b4"))

ax2.set_xlim(-4, 4.5)
ax2.set_ylim(-4, 4.5)
ax2.set_title(r"(b) SMR Conformal ETF Routing" + "\n" + r"Bounded Ellipsoids: $(h-\mu)^T (\Sigma+\lambda I)^{-1}(h-\mu) \leq \hat{q}_\alpha$", fontsize=11, fontweight='bold', color='#22543d')
ax2.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
os.makedirs("manuscript/figures", exist_ok=True)
plt.savefig("manuscript/figures/fig2_cones_vs_voronoi.pdf", bbox_inches='tight', dpi=300)
plt.savefig("manuscript/figures/fig2_cones_vs_voronoi.png", bbox_inches='tight', dpi=300)
plt.close()

print("Generated Figure 2: Polyhedral Cones vs Compact Voronoi Cells successfully!")
