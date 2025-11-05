import numpy as np
import matplotlib.pyplot as plt

np.random.seed(40)
SAMPLE_SIZE = 100000
BOX_BOUND = 2.5

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


def uniform_random_sample_cube(size, num_samples):
    """
    Uniformly samples num_samples points (x,y,z) in a cube with measurements [size]x[size]x[size].
    """
    return (
        np.random.uniform(-0.5 * size, 0.5 * size, num_samples),
        np.random.uniform(-0.5 * size, 0.5 * size, num_samples),
        np.random.uniform(-0.5 * size, 0.5 * size, num_samples)
    )


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
    sample_seeds = deterministic_sample_single_sequence(3, seed, m)
    x = deterministic_sample_single_sequence(num_samples, sample_seeds[0], m)
    y = deterministic_sample_single_sequence(num_samples, sample_seeds[1], m)
    z = deterministic_sample_single_sequence(num_samples, sample_seeds[2], m)

    x = x * BOX_BOUND - BOX_BOUND/2
    y = y * BOX_BOUND - BOX_BOUND/2
    z = z * BOX_BOUND - BOX_BOUND/2
    return (x, y, z)


def points_in_intersection(samples, k, r, R, x_0=0, y_0=0, z_0=0):
    x_array, y_array, z_array = samples
    """
    Checks for points (x,y,z) if they exist in the intersection between a sphere with radius
    k and a torus with minor radius r, major radius R, optionally off-center.
    
    Returns number of points in intersection, and the x, y and z arrays containing those points.
    """
    mask = is_in_sphere(x_array, y_array, z_array, k) & is_in_torus(x_array, y_array, z_array, r, R, x_0, y_0, z_0)
    intersection = (x_array[mask], y_array[mask], z_array[mask])
    non_intersection = (x_array[~mask], y_array[~mask], z_array[~mask])
    return intersection, non_intersection


def plot_intersection(x, y, z, sample_fraction=1, fig=None):
    """
    Displays points in 3D space using a scatter plot. If sample_fraction is passed and not 1,
    the defined fraction will be sampled from the points for plotting.
    """
    if sample_fraction != 1:
        idx = np.random.choice(len(x), int(len(x) * sample_fraction), replace=False)
        x, y, z = x[idx], y[idx], z[idx]
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')


def estimate_volume(intersection, box_bound):
    x_in, _, _ = intersection
    """
    Estimates the volume of the intersection between a sphere with radius k and a torus with
    minor radius r, major radius R using Monte Carlo simulation and random uniform sampling.
    """
    in_box = SAMPLE_SIZE
    estimated_volume = (len(x_in) / in_box) * box_bound**3
    return estimated_volume


print("ASSIGNMENT 1.1")
samples = uniform_random_sample_cube(BOX_BOUND, SAMPLE_SIZE)
intersection, nonintersection = points_in_intersection(samples, 1, 0.4, 0.75)
volume = estimate_volume(intersection, BOX_BOUND)
print(f"Case a: k = 1, R = 0.75 and r = 0.4, ESTIMATED AREA: {volume}")
samples = uniform_random_sample_cube(BOX_BOUND, SAMPLE_SIZE)
intersection, nonintersection = points_in_intersection(samples, 1, 0.5, 0.5)
volume = estimate_volume(intersection, BOX_BOUND)
print(f"Case b: k = 1, R = 0.5 and r = 0.5, ESTIMATED AREA: {volume}")


print("ASSIGNMENT 1.2")
samples = deterministic_sample(SAMPLE_SIZE, BOX_BOUND, seed=0.1)
intersection, nonintersection = points_in_intersection(samples, 1, 0.4, 0.75)
volume = estimate_volume(intersection, BOX_BOUND)
print(f"Case a: k = 1, R = 0.75 and r = 0.4, ESTIMATED AREA: {volume}")
samples = deterministic_sample(SAMPLE_SIZE, BOX_BOUND, seed=0.1)
intersection, nonintersection = points_in_intersection(samples, 1, 0.5, 0.5)
volume = estimate_volume(intersection, BOX_BOUND)
print(f"Case b: k = 1, R = 0.5 and r = 0.5, ESTIMATED AREA: {volume}")


print("ASSIGNMENT 1.3a")
samples = uniform_random_sample_cube(BOX_BOUND, SAMPLE_SIZE)
intersection, nonintersection = points_in_intersection(samples, 1, 0.4, 0.75, z_0=0.1)
volume = estimate_volume(intersection, BOX_BOUND)
print(f"Case: k = 1, R = 0.75, r = 0.4 and z_0 = 0.1, ESTIMATED AREA: {volume}")

