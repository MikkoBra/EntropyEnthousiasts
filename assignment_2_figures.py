import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

fig_folder = "a2_figures"


### SIMULATOR ASSUMPTIONS ###

# Distribution of arrival time deltas (single simulation)
save_path = os.path.join("a2_data", "snapshots_test.npy")
snapshots = np.load(save_path, allow_pickle=True)
single_snapshot = snapshots[0]
start_times = single_snapshot["start_times"]
time_deltas = np.diff(start_times)
plt.figure(figsize=(6,6))
plt.hist(time_deltas, bins=50, alpha=0.7, color='skyblue', edgecolor='k')
plt.axvline(np.mean(time_deltas), color='red', linestyle='--', label=f'Mean $\Delta t$ ({np.mean(time_deltas):.2f})')
lambda_rate = 0.85
x = np.linspace(0, max(time_deltas), 500)
pdf = lambda_rate * np.exp(-lambda_rate * x)
bin_width = (max(time_deltas) - min(time_deltas)) / 50
pdf_scaled = pdf * len(time_deltas) * bin_width
plt.plot(x, pdf_scaled, color='green', linewidth=2, label=f'Exponential (λ={lambda_rate})')
plt.xlabel('Time Between Arrivals')
plt.ylabel('Frequency')
plt.title('Distribution of Arrival Time Deltas (One Simulation)')
plt.legend()
plt.savefig(os.path.join(fig_folder, '2_arrival_dist.png'), dpi=300)
plt.close()

# Distribution of service times (test simulation)
service_times = single_snapshot["server_times"]
plt.figure(figsize=(6,6))
plt.hist(service_times, bins=50, alpha=0.7, color='skyblue', edgecolor='k')
plt.axvline(np.mean(service_times), color='red', linestyle='--', label=f'Mean ({np.mean(service_times):.2f})')
x = np.linspace(min(service_times), max(service_times), 500)
pdf = stats.norm.pdf(x, loc=1.0, scale=0.25)
bin_width = (max(service_times) - min(service_times)) / 50
pdf_scaled = pdf * len(service_times) * bin_width
plt.plot(x, pdf_scaled, color='green', linewidth=2, label='Normal(1, 0.25)')
plt.xlabel('Service Time')
plt.ylabel('Frequency')
plt.title('Distribution of Service Times (One Simulation, Test Scenario)')
plt.legend()
plt.savefig(os.path.join(fig_folder, '2_service_dist.png'), dpi=300)
plt.close()


### T-TEST ASSUMPTIONS ###

save_path = os.path.join("a2_data", "mean_queue_times.npy")
waiting_times = np.load(save_path)

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

save_path = os.path.join("a2_data", "bootstrap.npy")
bootstrap = np.load(save_path)

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

save_path = os.path.join("a2_data", "two_sample_bootstrap.npy")
two_sample_boot = np.load(save_path)

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


### INTERVENTION RESULTS ###

# Mean queue time over simulations (base simulation)
save_path = os.path.join("a2_data", "snapshots_baseline parameters.npy")
snapshots = np.load(save_path, allow_pickle=True)
queue_matrix = np.array([snap["queue_times"] for snap in snapshots])
mean_queue = np.mean(queue_matrix, axis=0)
std_queue  = np.std(queue_matrix, axis=0)
passengers = snapshots[0]["passengers"]
plt.figure(figsize=(6, 6))
plt.fill_between(
    passengers,
    mean_queue - std_queue,
    mean_queue + std_queue,
    alpha=0.3,
    color='skyblue',
    label='Variance region'
)
plt.plot(
    passengers,
    mean_queue,
    color='red',
    linewidth=2,
    label='Mean queue time'
)
plt.xlabel('Passenger Index')
plt.ylabel('Queue Time')
plt.title(f'Mean Queue Time Over Passengers with Variance Region ($\sigma$)\nBase Parameters')
plt.legend()
plt.savefig(os.path.join(fig_folder, '2_queue_base.png'), dpi=300)
plt.close()

# Mean queue time over simulations (base simulation, first 25 passengers)
N = 25
small_mean_queue = mean_queue[:N]
small_std_queue = std_queue[:N]
small_passengers = passengers[:N]
plt.figure(figsize=(6, 6))
plt.fill_between(
    small_passengers,
    small_mean_queue - small_std_queue,
    small_mean_queue + small_std_queue,
    alpha=0.3,
    color='skyblue',
    label='Variance region'
)
plt.plot(
    small_passengers,
    small_mean_queue,
    color='red',
    linewidth=2,
    label='Mean queue time'
)
plt.xlabel('Passenger Index')
plt.ylabel('Queue Time')
plt.title(f'Mean Queue Time Over First 25 Passengers with Variance Region ($\sigma$)')
plt.legend()
plt.savefig(os.path.join(fig_folder, '2_queue_base_25.png'), dpi=300)
plt.close()

# Mean queue time over simulations (intervention A)
save_path = os.path.join("a2_data", "snapshots_intervention A.npy")
snapshots = np.load(save_path, allow_pickle=True)
queue_matrix = np.array([snap["queue_times"] for snap in snapshots])
mean_queue = np.mean(queue_matrix, axis=0)
std_queue  = np.std(queue_matrix, axis=0)
passengers = snapshots[0]["passengers"]
plt.figure(figsize=(6, 6))
plt.fill_between(
    passengers,
    mean_queue - std_queue,
    mean_queue + std_queue,
    alpha=0.3,
    color='skyblue',
    label='Variance region'
)
plt.plot(
    passengers,
    mean_queue,
    color='red',
    linewidth=2,
    label='Mean queue time'
)
plt.xlabel('Passenger Index')
plt.ylabel('Queue Time')
plt.title(f'Mean Queue Time Over Passengers with Variance Region ($\sigma$)\nIntervention A')
plt.legend()
plt.savefig(os.path.join(fig_folder, '2_queue_int_A.png'), dpi=300)
plt.close()

# Mean queue time over simulations (intervention B)
save_path = os.path.join("a2_data", "snapshots_intervention B.npy")
snapshots = np.load(save_path, allow_pickle=True)
queue_matrix = np.array([snap["queue_times"] for snap in snapshots])
mean_queue = np.mean(queue_matrix, axis=0)
std_queue  = np.std(queue_matrix, axis=0)
passengers = snapshots[0]["passengers"]
plt.figure(figsize=(6, 6))
plt.fill_between(
    passengers,
    mean_queue - std_queue,
    mean_queue + std_queue,
    alpha=0.3,
    color='skyblue',
    label='Variance region'
)
plt.plot(
    passengers,
    mean_queue,
    color='red',
    linewidth=2,
    label='Mean queue time'
)
plt.xlabel('Passenger Index')
plt.ylabel('Queue Time')
plt.title(f'Mean Queue Time Over Passengers with Variance Region ($\sigma$)\nIntervention B')
plt.legend()
plt.savefig(os.path.join(fig_folder, '2_queue_int_B.png'), dpi=300)
plt.close()

# Mean queue time over simulations (bonus intervention)
save_path = os.path.join("a2_data", "snapshots_bonus parameters.npy")
snapshots = np.load(save_path, allow_pickle=True)
queue_matrix = np.array([snap["queue_times"] for snap in snapshots])
mean_queue = np.mean(queue_matrix, axis=0)
std_queue  = np.std(queue_matrix, axis=0)
passengers = snapshots[0]["passengers"]
plt.figure(figsize=(6, 6))
plt.fill_between(
    passengers,
    np.maximum(mean_queue - std_queue, 0),
    mean_queue + std_queue,
    alpha=0.3,
    color='skyblue',
    label='Variance region'
)
plt.plot(
    passengers,
    mean_queue,
    color='red',
    linewidth=2,
    label='Mean queue time'
)
plt.xlabel('Passenger Index')
plt.ylabel('Queue Time')
plt.title(f'Mean Queue Time Over Passengers with Variance Region ($\sigma$)\nBonus Intervention')
plt.legend()
plt.savefig(os.path.join(fig_folder, '2_queue_int_bonus.png'), dpi=300)
plt.close()