import numpy as np
import matplotlib.pyplot as plt

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# Parameters
t_start = 0.3
gamma_max = 1.0
alpha = 2.0  # controls sharpness of increase

# Time array from 1 to 0 (reverse direction as requested)
t = np.linspace(1, 0, 1000)

# Exponential triggering schedule function
def gamma_t(t, t_start, gamma_max, alpha):
    """Calculate gamma(t) based on the exponential triggering schedule"""
    indicator = (t <= t_start).astype(float)
    return gamma_max * (1 - t / t_start) ** alpha * indicator

# Calculate gamma values
gamma_values = gamma_t(t, t_start, gamma_max, alpha)

# Create the plot
plt.figure(figsize=(5, 3))
plt.plot(t, gamma_values, 'b-', linewidth=2, label=f'γ(t), α={alpha}')
plt.axvline(x=t_start, color='r', linestyle='--', alpha=0.7, label=f't_start = {t_start}')
plt.xlabel('Time (t)', fontname='Times New Roman', fontsize=14)
plt.ylabel('γ(t)', fontname='Times New Roman', fontsize=14)
plt.title('Exponential Triggering Schedule: γ(t) vs Time', fontname='Times New Roman', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(prop={'family': 'Times New Roman'})

# Highlight key regions
plt.axvspan(0, t_start, alpha=0.1, color='green', label='Constraints Active')
plt.axvspan(t_start, 1, alpha=0.1, color='red', label='Constraints Inactive')

plt.legend(prop={'family': 'Times New Roman'})
plt.xlim(1, 0)  # Reverse x-axis as requested (from 1 to 0)
plt.ylim(-0.05, 1.05)
plt.tight_layout()

plt.savefig('time_dependent_weight_curve.pdf')
plt.show()
plt.close()