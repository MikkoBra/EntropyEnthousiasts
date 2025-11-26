import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

save_path = os.path.join("a2_data", "rep_Wq.npy")
waiting_times = np.load(save_path)

### T-TEST ASSUMPTIONS ###
fig_folder = "a2_figures"

# Histogram
plt.figure(figsize=(8,8))
plt.hist(waiting_times, bins=12, alpha=0.7, color='skyblue', edgecolor='k')
plt.axvline(np.mean(waiting_times), color='red', linestyle='--', label='Mean')
plt.xlabel('Replication mean Wq')
plt.ylabel('Frequency')
plt.title('Histogram of replication means')
plt.legend()
plt.savefig(os.path.join(fig_folder, '2a_means_histogram.png'), dpi=300)
plt.close()

# Q-Q Plot
stats.probplot(waiting_times, dist="norm", plot=plt)
output_path = os.path.join(fig_folder, "2a_QQ_plot.png")
plt.title("QQ Plot for normality")
plt.savefig(output_path, dpi=300)
plt.close()

### WILCOXON ASSUMPTIONS ###

theoretical_Wq = 3.0104
diffs = waiting_times - theoretical_Wq

# 1. Histogram of differences
plt.figure(figsize=(8,6))
plt.hist(diffs, bins=12, alpha=0.7, color='lightgreen', edgecolor='k')
plt.axvline(np.median(diffs), color='red', linestyle='--', label='Median')
plt.xlabel('Difference (Simulated - Theoretical Wq)')
plt.ylabel('Frequency')
plt.title('Histogram of differences from theoretical Wq')
plt.legend()
plt.savefig(os.path.join(fig_folder, '2a_diff_histogram.png'), dpi=300)
plt.close()

# 2. Boxplot of differences
plt.figure(figsize=(6,6))
plt.boxplot(diffs, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.ylabel('Difference (Simulated - Theoretical Wq)')
plt.title('Boxplot of differences from theoretical Wq')
plt.savefig(os.path.join(fig_folder, '2a_diff_boxplot.png'), dpi=300)
plt.close()

# 3. Q-Q plot of differences
plt.figure(figsize=(6,6))
stats.probplot(diffs, dist="norm", plot=plt)
plt.title('Q-Q Plot of differences for symmetry')
plt.savefig(os.path.join(fig_folder, '2a_diff_QQ_plot.png'), dpi=300)
plt.close()