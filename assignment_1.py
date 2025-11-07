import numpy as np
import matplotlib.pyplot as plt

np.random.seed(40)
SAMPLE_SIZE = 100000            # Number of samples for Monte Carlo estimation
NUM_ESTIMATIONS = 100           # Number of iterations of Monte Carlo to average the estimation over
BOX_BOUND = 3                   # Side length of the bounding box
SMALL_BOX_BOUND = 1.5           # Side length of the smaller bounding box
BOX_PROBABILITY = 0.5           # Probability to sample from the big bounding box in mixed sampling
FIND_OPTIMAL_BOX_P = False      # Find optimal big box sampling probability using mean within-run error
DETERMINISTIC_SEED = 0.7        # Seed for the logistic map sampling method
SAVE_FIGURES = False            # Save figures to local files


################################# CONDITIONALS AND SAMPLING #################################

def is_in_sphere(x, y, z, k):
    """
    Conditional function that checks if a point (x,y,z) is within a sphere with radius k.
    """
    return x**2 + y**2 + z**2 <= k**2


def is_in_torus(x, y, z, r, R, x_0, y_0, z_0):
    """
    Conditional function that checks if a point (x,y,z) is within a torus with minor radius r,
    major radius R.
    """
    return (np.sqrt((x-x_0)**2 + (y-y_0)**2) - R)**2 + (z-z_0)**2 <= r**2


def uniform_random_sample_cube(size, z_center, num_samples):
    """
    Uniformly samples num_samples points (x,y,z) in a cube with measurements [size]x[size]x[size].
    """
    return (
        np.random.uniform(-0.5 * size, 0.5 * size, num_samples),
        np.random.uniform(-0.5 * size, 0.5 * size, num_samples),
        np.random.uniform((-0.5 + z_center) * size, (0.5 + z_center) * size, num_samples)
    )


def mixture_sample_cube(p_B, num_samples, box_bound, small_box_bound):
    """
    Uniformly samples within two bounding boxes; one bigger box centered at 0, one
    smaller box centered at 0.1. The probability of sampling from the bigger box is
    defined by p_B.
    """
    if p_B == 1.0:
        return uniform_random_sample_cube(box_bound, 0, num_samples)
    elif p_B == 0:
        return uniform_random_sample_cube(small_box_bound, 0.1, num_samples)
    num_B_samples = int(num_samples * p_B)
    x_B, y_B, z_B = uniform_random_sample_cube(box_bound, 0, num_B_samples)
    x_S, y_S, z_S = uniform_random_sample_cube(small_box_bound, 0.1, num_samples - num_B_samples)
    x_total = np.concatenate([x_B, x_S])
    y_total = np.concatenate([y_B, y_S])
    z_total = np.concatenate([z_B, z_S])
    return (x_total, y_total, z_total)


def deterministic_sample_single_sequence(num_samples, x_0, m):
    """
    Creates a sequence of values using deterministic function X_i+1 = m*X_i(1-X_i).
    """
    samples = np.zeros(num_samples)
    samples[0] = x_0
    for i in range(1, num_samples):
        samples[i] = m*samples[i-1]*(1-samples[i-1])
    return samples


def deterministic_sample(num_samples, box, m=3.8, seed=0.5):
    x = deterministic_sample_single_sequence(num_samples, seed, m)
    y = deterministic_sample_single_sequence(num_samples, seed + 0.01, m)
    z = deterministic_sample_single_sequence(num_samples, seed - 0.01, m)

    x = BOX_BOUND * (x - 0.5)
    y = BOX_BOUND * (y - 0.5)
    z = BOX_BOUND * (z - 0.5)

    return (x, y, z)


################################# ANALYSIS #################################

def points_in_intersection(samples, k, r, R, x_0=0, y_0=0, z_0=0):
    """
    Checks for points (x,y,z) if they exist in the intersection between a sphere with radius
    k and a torus with minor radius r, major radius R, optionally off-center.
    
    Returns number of points in intersection, and the x, y and z arrays containing those points.
    """
    x_array, y_array, z_array = samples
    mask = is_in_sphere(x_array, y_array, z_array, k) &\
            is_in_torus(x_array, y_array, z_array, r, R, x_0, y_0, z_0)
    intersection = (x_array[mask], y_array[mask], z_array[mask])
    non_intersection = (x_array[~mask], y_array[~mask], z_array[~mask])
    return intersection, non_intersection


def error(intersection, p_B=1):
    """
    Calculates error of estimated volume using a binomial distribution.
    """
    x_in, _, _ = intersection
    probability_in = len(x_in) / SAMPLE_SIZE
    std = np.sqrt(probability_in * (1 - probability_in))
    return (
        ((p_B * (BOX_BOUND ** 3))
        + ((1 - p_B) * (BOX_BOUND ** 3)))
        * std / np.sqrt(SAMPLE_SIZE)
    )


def estimate_volume(intersection, box_bound, small_box_bound=0, p_B=1):
    """
    Estimates the volume of the intersection between a sphere with radius k and a torus with
    minor radius r, major radius R using Monte Carlo simulation and random uniform sampling.
    """
    x_in, _, _ = intersection
    in_box = SAMPLE_SIZE
    estimated_volume = (
        p_B * (len(x_in) / SAMPLE_SIZE) * box_bound**3
        + (1 - p_B) * small_box_bound**3 * (len(x_in) / SAMPLE_SIZE)
    )
    return estimated_volume


def multiple_uniform_runs(k, r, R, z_0=0):
    """
    Runs multiple Monte Carlo estimations based on uniform sampling of a single bounding
    box. Returns the mean estimate, the standard error between runs, and the mean
    within-run error.
    """
    estimates = np.zeros(NUM_ESTIMATIONS)
    errors = np.zeros(NUM_ESTIMATIONS)
    print(f"Performing {NUM_ESTIMATIONS} Monte Carlo estimations...")
    for i in range(NUM_ESTIMATIONS):
        samples = uniform_random_sample_cube(BOX_BOUND, 0, SAMPLE_SIZE)
        intersection, nonintersection = points_in_intersection(samples, k, r, R, z_0=z_0)
        volume = estimate_volume(intersection, BOX_BOUND)
        estimates[i] = volume
        errors[i] = error(intersection)
    mean_estimate = np.mean(estimates)
    mean_error = np.mean(errors)
    std_between_runs = np.std(estimates, ddof=1)
    se = std_between_runs / np.sqrt(NUM_ESTIMATIONS)
    figname = f"Uniform samples, k = {k}, R = {R}, r = {r}"
    if z_0 > 0:
        figname += f", z_0 = {z_0}"
    create_figure(samples, intersection, nonintersection, estimates, figname)
    return mean_estimate, se, mean_error


def multiple_deterministic_runs(k, r, R):
    """
    Runs multiple Monte Carlo estimations based on deterministic sampling of a
    single bounding box. Returns the mean estimate, the standard error between
    runs, and the mean within-run error.
    """
    estimates = np.zeros(NUM_ESTIMATIONS)
    errors = np.zeros(NUM_ESTIMATIONS)
    seed = DETERMINISTIC_SEED
    print(f"Performing {NUM_ESTIMATIONS} Monte Carlo estimations...")
    for i in range(NUM_ESTIMATIONS):
        samples = deterministic_sample(SAMPLE_SIZE, BOX_BOUND, seed=seed)
        intersection, nonintersection = points_in_intersection(samples, k, r, R)
        volume = estimate_volume(intersection, BOX_BOUND)
        estimates[i] = volume
        errors[i] = error(intersection)
        seed += 0.0001
    mean_estimate = np.mean(estimates)
    mean_error = np.mean(errors)
    std_between_runs = np.std(estimates, ddof=1)
    se = std_between_runs / np.sqrt(NUM_ESTIMATIONS)
    figname = f"Deterministic samples, k = {k}, R = {R}, r = {r}"
    create_figure(samples, intersection, nonintersection, estimates, figname)
    return mean_estimate, se, mean_error
    

def optimal_box_distribution(probs, num_estimations=10):
    """
    Finds the big bounding box sampling probability that yields the lowest mean
    within-run error.
    """
    best_error = np.inf
    best_probability = 0
    mean_errors = np.zeros_like(probs)
    for i, probability in enumerate(probs):
        errors = np.zeros(num_estimations)
        for j in range(num_estimations):
            samples = mixture_sample_cube(probability, SAMPLE_SIZE, BOX_BOUND, SMALL_BOX_BOUND)
            intersection, _ = points_in_intersection(samples, 1, 0.4, 0.75, z_0=0.1)
            errors[j] = error(intersection)
        if np.mean(errors) < best_error:
            best_error = np.mean(errors)
            best_probability = probability
        mean_errors[i] = np.mean(errors)
    return best_probability, mean_errors


################################# PLOTS #################################

def create_figure(samples, intersection, nonintersection, estimates, name=None):
    fig = plt.figure(figsize=(20,10))
    fig.suptitle(name, fontsize=20)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_samples(samples, intersection, nonintersection, ax=ax1)
    ax2 = fig.add_subplot(1, 2, 2)
    plot_means(estimates, ax=ax2)
    if SAVE_FIGURES:
        plt.savefig(f"figures/{name}.png")
    else:
        plt.show()
        plt.close()


def plot_samples(samples, intersection, nonintersection, fig=None, intersection_sample_fraction=0.1, nonintersection_sample_fraction=0.1, ax=None):
    samples = np.array(samples)
    intersection = np.array(intersection)
    nonintersection = np.array(nonintersection)

    intersection_idx = np.random.choice(len(intersection[0]), int(len(intersection[0]) * intersection_sample_fraction), replace=False)
    nonintersection_idx = np.random.choice(len(nonintersection[0]), int(len(nonintersection[0]) * nonintersection_sample_fraction), replace=False)

    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
    ax.scatter(intersection[0, intersection_idx], intersection[1, intersection_idx], intersection[2, intersection_idx], c='g', alpha=0.5, label="Hit")
    ax.scatter(nonintersection[0, nonintersection_idx], nonintersection[1, nonintersection_idx], nonintersection[2, nonintersection_idx], c='r', alpha=0.1, label="Miss")
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$y$', fontsize=16)
    ax.set_zlabel(r'$z$', fontsize=16)
    ax.set_xlim(-BOX_BOUND/2, BOX_BOUND/2)
    ax.set_ylim(-BOX_BOUND/2, BOX_BOUND/2)
    ax.set_zlim(-BOX_BOUND/2, BOX_BOUND/2)
    ax.legend(fontsize=16)
    ax.set_title("Samples", fontsize=18)
    ax.view_init(elev=45, azim=-45)

def plot_means(estimates, ax=None):
    means = np.zeros(NUM_ESTIMATIONS)
    stderrs = np.zeros(NUM_ESTIMATIONS)
    for i in range(NUM_ESTIMATIONS):
        if i == 0:
            means[i] = estimates[i]
            continue
        means[i] = np.mean(estimates[:i])
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot()
    ax.plot(np.arange(len(means)), means, 'b', label="Mean volume estimate")
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Mean volume estimate", fontsize=16)
    ax.set_title("Mean volume estimate over iterations", fontsize=18)


################################# RESULTS #################################

def exercise_1_1(results_array):
    print("ASSIGNMENT 1.1")
    print(f"Case: k = 1, R = 0.75, and r = 0.4, 100 runs")
    volume, se, mean_error = multiple_uniform_runs(1, 0.4, 0.75)
    std_between_runs = se * np.sqrt(NUM_ESTIMATIONS)
    print(f"ESTIMATED VOLUME: {volume:.3f}\n" +
          f"STANDARD ERROR: {se:.3f}\nMEAN ERROR: {mean_error:.3f}, ESTIMATE STD: {std_between_runs:.3f}\n")
    results_array["1.1 Case 1: "] = volume
    print(f"Case: k = 1, R = 0.5, and r = 0.5, 100 runs")
    volume, se, mean_error = multiple_uniform_runs(1, 0.5, 0.5)
    std_between_runs = se * np.sqrt(NUM_ESTIMATIONS)
    print(f"ESTIMATED VOLUME: {volume:.3f}\n" +
          f"STANDARD ERROR: {se:.3f}\nMEAN ERROR: {mean_error:.3f}, ESTIMATE STD: {std_between_runs:.3f}\n")
    results_array["1.1 Case 2: "] = volume


def exercise_1_2(results_array):
    print("ASSIGNMENT 1.2")
    deterministic_rng_test()
    print(f"Case: k = 1, R = 0.75, r = 0.4, 100 runs")
    volume, se, mean_error = multiple_deterministic_runs(1, 0.4, 0.75)
    std_between_runs = se * np.sqrt(NUM_ESTIMATIONS)
    print(f"ESTIMATED VOLUME: {volume:.3f}\n" +
          f"STANDARD ERROR: {se:.3f}\nMEAN ERROR: {mean_error:.3f}, ESTIMATE STD: {std_between_runs:.3f}\n")
    results_array["1.2 Case 1: "] = volume
    print(f"Case: k = 1, R = 0.5, r = 0.5, 100 runs")
    volume, se, mean_error = multiple_deterministic_runs(1, 0.5, 0.5)
    std_between_runs = se * np.sqrt(NUM_ESTIMATIONS)
    print(f"ESTIMATED VOLUME: {volume}\n" +
          f"STANDARD ERROR: {se:.3f}\nMEAN ERROR: {mean_error:.3f}, ESTIMATE STD: {std_between_runs:.3f}\n")
    results_array["1.2 Case 2: "] = volume


def exercise_1_3(results_array):
    print("ASSIGNMENT 1.3a")
    print(f"Case: k = 1, R = 0.75, r = 0.4 and z_0 = 0.1, 100 runs")
    volume, se, mean_error = multiple_uniform_runs(1, 0.4, 0.75, z_0=0.1)
    std_between_runs = se * np.sqrt(NUM_ESTIMATIONS)
    print(f"ESTIMATED VOLUME: {volume}\n" +
          f"STANDARD ERROR: {se:.3f}\nMEAN ERROR: {mean_error:.3f}, ESTIMATE STD: {std_between_runs:.3f}\n")
    results_array["1.3 Case 1: "] = volume
    print("ASSIGNMENT 1.3b")
    print(f"Case: k = 1, R = 0.75, r = 0.4 and z_0 = 0.1, 100 runs")
    if FIND_OPTIMAL_BOX_P:
        ps_to_evaluate = np.linspace(0, 1, 21)
        print("CALCULATING OPTIMAL LARGE BOUNDING BOX SAMPLE PROBABILITY p:")
        p_B, mean_errors = optimal_box_distribution(ps_to_evaluate)
        print("STANDARD MEAN ERROR PER p:")
        for i, mean_error in enumerate(mean_errors):
            print(f"{ps_to_evaluate[i]:.2f}:\t{mean_error:.4f}")
        print(f"OPTIMAL LARGE BOUNDING BOX SAMPLE PROBABILITY: {p_B}")
    else:
        p_B = BOX_PROBABILITY
    estimates = np.zeros(NUM_ESTIMATIONS)
    errors = np.zeros(NUM_ESTIMATIONS)
    print(f"Performing {NUM_ESTIMATIONS} Monte Carlo estimations...")
    for i in range(NUM_ESTIMATIONS):
        samples = mixture_sample_cube(p_B, SAMPLE_SIZE, BOX_BOUND, SMALL_BOX_BOUND)
        intersection, nonintersection = points_in_intersection(samples, 1, 0.4, 0.75, z_0=0.1)
        volume = estimate_volume(intersection, BOX_BOUND, SMALL_BOX_BOUND, p_B)
        estimates[i] = volume
        errors[i] = error(intersection, p_B)
    mean_estimate = np.mean(estimates)
    mean_error = np.mean(errors)
    std_between_runs = np.std(estimates, ddof=1)
    se = std_between_runs / np.sqrt(NUM_ESTIMATIONS)
    print(f"ESTIMATED VOLUME: {mean_estimate:.3f}\n" +
          f"STANDARD ERROR: {se:.3f}\nMEAN ERROR: {mean_error:.3f}, ESTIMATE STD: {std_between_runs:.3f}\n")
    results_array["1.3 Case 1 mixed: "] = mean_estimate
    

def deterministic_rng_test():
    """
    Plots the distribution of the samples from the deterministic sampling method.
    """
    sample = deterministic_sample_single_sequence(10000, x_0=0.5, m=3.8)
    expectation = np.mean(sample)
    plt.figure()
    plt.hist(sample, bins=100)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.axvline(expectation, linestyle='dotted', color='k', label='Expected value')
    plt.title('Deterministic pRNG Test')
    plt.legend()
    if SAVE_FIGURES:
        plt.savefig('figures/deterministic_rng_test.png')
    else:
        plt.show()
        plt.close()
    print(f'Deterministic pRNG Expectation: {expectation:.3f}\n')

results = {}
exercise_1_1(results)
exercise_1_2(results)
exercise_1_3(results)
print("##################################\n\tESTIMATES OVERVIEW\n##################################")
for key, item in results.items():
    print(f"{key}{item:.4}")
