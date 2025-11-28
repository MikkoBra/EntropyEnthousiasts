import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

save_path = os.path.join("a2_data", "mean_queue_times.npy")
waiting_times = np.load(save_path)
save_path = os.path.join("a2_data", "bootstrap.npy")
bootstrap = np.load(save_path)
save_path = os.path.join("a2_data", "two_sample_bootstrap.npy")
two_sample_boot = np.load(save_path)

### T-TEST ASSUMPTIONS ###
fig_folder = "a2_figures"

# 1. Histogram
plt.figure(figsize=(6,6))
plt.hist(waiting_times, bins=12, alpha=0.7, color='skyblue', edgecolor='k')
plt.axvline(np.mean(waiting_times), color='red', linestyle='--', label=f'Mean ({np.mean(waiting_times):.2f},'+r' $W_{q\_theory}$=3.01)')
plt.xlabel('Replication mean Wq')
plt.ylabel('Frequency')
plt.title('Distribution of Sample Mean Waiting Times')
plt.legend()
plt.savefig(os.path.join(fig_folder, '2a_means_dist.png'), dpi=300)
plt.close()

# 2. Boxplot
plt.figure(figsize=(6,6))
plt.boxplot(waiting_times, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.ylabel('Mean Queue Time')
plt.title('Boxplot of Sample Mean Waiting Times')
plt.savefig(os.path.join(fig_folder, '2a_means_box.png'), dpi=300)
plt.close()

# 3. Q-Q Plot
stats.probplot(waiting_times, dist="norm", plot=plt)
output_path = os.path.join(fig_folder, "2a_means_qq.png")
plt.title("QQ Plot for Normality of Sample Means")
plt.savefig(output_path, dpi=300)
plt.close()

### WILCOXON ASSUMPTIONS ###

theoretical_Wq = 3.0104
diffs = waiting_times - theoretical_Wq

# 1. Histogram of differences
plt.figure(figsize=(6,6))
plt.hist(diffs, bins=12, alpha=0.7, color='lightgreen', edgecolor='k')
plt.axvline(np.median(diffs), color='red', linestyle='--', label='Median')
plt.xlabel('Difference (Simulated - Theoretical Wq)')
plt.ylabel('Frequency')
plt.title('Distribution of Differences from Theoretical Wq')
plt.legend()
plt.savefig(os.path.join(fig_folder, '2a_diffs_dist.png'), dpi=300)
plt.close()

# 2. Boxplot of differences
plt.figure(figsize=(6,6))
plt.boxplot(diffs, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.ylabel('Difference (Simulated - Theoretical Wq)')
plt.title('Boxplot of Differences from Theoretical Wq')
plt.savefig(os.path.join(fig_folder, '2a_diffs_box.png'), dpi=300)
plt.close()


### BOOTSTRAP ###

# One-sample
plt.figure(figsize=(6,6))
plt.hist(bootstrap, bins=20, alpha=0.7, color='lightgreen', edgecolor='k')
plt.axvline(np.mean(bootstrap), color='red', linestyle='--', label=f'Mean ({np.mean(bootstrap):.2f},'+r' $W_{q\_theory}$=3.01)')
plt.xlabel('Bootstrapped Sample Means')
plt.ylabel('Frequency')
plt.title('Histogram of Bootstrapped Sample Means')
plt.legend(fontsize=8, handlelength=1, handletextpad=0.4, markerscale=0.7)
plt.savefig(os.path.join(fig_folder, '2a_bootstrap.png'), dpi=300)
plt.close()

# Two-sample differences
plt.figure(figsize=(6,6))
mean_bs = np.mean(two_sample_boot)
plt.hist(two_sample_boot, bins=20, alpha=0.7, color='lightgreen', edgecolor='k')
plt.axvline(mean_bs, color='red', linestyle='--', label=f'Mean ({mean_bs:.2f})')
plt.xlabel('Bootstrapped Mean Differences (x - y)')
plt.ylabel('Frequency')
plt.title('Histogram of Two-sample Bootstrapped Mean Differences')
plt.legend(fontsize=8, handlelength=1, handletextpad=0.4, markerscale=0.7)
plt.savefig(os.path.join(fig_folder, '2a_two_sample_boot_diffs.png'), dpi=300)
plt.close()