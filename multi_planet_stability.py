import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

G = 6.67430e-11
M_SUN = 1.989e30
M_EARTH = 5.972e24
AU = 1.496e11
YEAR = 365.25 * 24 * 3600

# CUDA kernel for calculating accelerations
cuda_code = """
__global__ void calculate_acceleration(float *pos, float *acc, float *masses, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float ax = 0.0f, ay = 0.0f;
        float xi = pos[i*2], yi = pos[i*2+1];
        float mi = masses[i];

        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                float dx = pos[j*2] - xi;
                float dy = pos[j*2+1] - yi;
                float dist_sqr = dx*dx + dy*dy;
                float inv_dist_cube = rsqrtf(dist_sqr * dist_sqr * dist_sqr);
                float mj = masses[j];
                float fac = mj * inv_dist_cube;
                ax += fac * dx;
                ay += fac * dy;
            }
        }
        acc[i*2] = ax * 6.67430e-11;
        acc[i*2+1] = ay * 6.67430e-11;
    }
}
"""

# Compile CUDA kernel
mod = SourceModule(cuda_code)
calculate_acceleration_gpu = mod.get_function("calculate_acceleration")

def initialize_system(num_planets, perturbation_angle):
    # Initialize the Sun at the origin
    positions = np.zeros((num_planets + 1, 2))
    velocities = np.zeros((num_planets + 1, 2))
    masses = np.array([M_SUN] + [M_EARTH] * num_planets)

    # Initialize planets
    angles = np.linspace(0, 2*np.pi, num_planets, endpoint=False)
    positions[1:, 0] = AU * np.cos(angles)
    positions[1:, 1] = AU * np.sin(angles)

    # Calculate initial velocities for circular orbit
    v = np.sqrt(G * M_SUN / AU)
    velocities[1:, 0] = -v * np.sin(angles)
    velocities[1:, 1] = v * np.cos(angles)

    # Apply perturbation to the first planet
    perturbation = perturbation_angle * np.pi / 180  # Convert to radians
    positions[1] = np.array([AU * np.cos(perturbation), AU * np.sin(perturbation)])
    velocities[1] = np.array([-v * np.sin(perturbation), v * np.cos(perturbation)])

    return positions, velocities, masses

def leapfrog_step_gpu(pos_gpu, vel_gpu, acc_gpu, masses_gpu, dt, num_bodies):
    block_size = 256
    grid_size = (num_bodies + block_size - 1) // block_size

    # First half-kick
    vel_gpu += 0.5 * acc_gpu * dt

    # Drift
    pos_gpu += vel_gpu * dt

    # Calculate new accelerations
    calculate_acceleration_gpu(
        pos_gpu, acc_gpu, masses_gpu, np.int32(num_bodies),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )

    # Second half-kick
    vel_gpu += 0.5 * acc_gpu * dt

def forest_ruth_step_gpu(pos_gpu, vel_gpu, acc_gpu, masses_gpu, dt, num_bodies):
    # Forest-Ruth coefficients
    w0 = -np.power(2, 1/3) / (2 - np.power(2, 1/3))
    w1 = 1 / (2 - np.power(2, 1/3))
    c1 = w1 / 2
    c4 = c1
    c2 = (w0 + w1) / 2
    c3 = c2
    d1 = w1
    d2 = w0
    d3 = w1

    block_size = 256
    grid_size = (num_bodies + block_size - 1) // block_size

    # First stage
    pos_gpu += c1 * vel_gpu * dt
    calculate_acceleration_gpu(
        pos_gpu, acc_gpu, masses_gpu, np.int32(num_bodies),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    vel_gpu += d1 * acc_gpu * dt

    # Second stage
    pos_gpu += c2 * vel_gpu * dt
    calculate_acceleration_gpu(
        pos_gpu, acc_gpu, masses_gpu, np.int32(num_bodies),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    vel_gpu += d2 * acc_gpu * dt

    # Third stage
    pos_gpu += c3 * vel_gpu * dt
    calculate_acceleration_gpu(
        pos_gpu, acc_gpu, masses_gpu, np.int32(num_bodies),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    vel_gpu += d3 * acc_gpu * dt

    # Fourth stage
    pos_gpu += c4 * vel_gpu * dt

def simulate_gpu(num_planets, years, dt, perturbation_angle, use_forest_ruth=False):
    positions, velocities, masses = initialize_system(num_planets, perturbation_angle)
    num_steps = int(years * YEAR / dt)
    num_bodies = num_planets + 1
    max_deviation = 0

    # Transfer data to GPU
    pos_gpu = gpuarray.to_gpu(positions.astype(np.float32))
    vel_gpu = gpuarray.to_gpu(velocities.astype(np.float32))
    acc_gpu = gpuarray.zeros_like(pos_gpu)
    masses_gpu = gpuarray.to_gpu(masses.astype(np.float32))

    initial_angles = np.arctan2(positions[1:, 1], positions[1:, 0])

    for _ in range(num_steps):
        if use_forest_ruth:
            forest_ruth_step_gpu(pos_gpu, vel_gpu, acc_gpu, masses_gpu, dt, num_bodies)
        else:
            leapfrog_step_gpu(pos_gpu, vel_gpu, acc_gpu, masses_gpu, dt, num_bodies)

        if _ % 100 == 0:  # Check deviation every 100 steps
            positions = pos_gpu.get()
            current_angles = np.arctan2(positions[1:, 1], positions[1:, 0])
            deviation = np.abs(current_angles - initial_angles)
            deviation = np.minimum(deviation, 2*np.pi - deviation)
            max_deviation = max(max_deviation, np.max(deviation))

    return max_deviation * 180 / np.pi  # Convert to degrees

def run_simulations(use_forest_ruth=False):
    planet_numbers = range(2, 201)
    stability_measures = []

    for N in planet_numbers:
        print(f"Simulating {N} planets...")
        max_deviation = simulate_gpu(N,
                                     years=1000,
                                     dt=1*24*3600,
                                     perturbation_angle=0.1,
                                     use_forest_ruth=use_forest_ruth)
        stability_measures.append(max_deviation)

    return planet_numbers, stability_measures

def plot_results(planet_numbers, stability_measures, method_name):
    plt.figure(figsize=(12, 8))
    plt.plot(planet_numbers, stability_measures, '-o')
    plt.xlabel('Number of Planets')
    plt.ylabel('Maximum Angular Deviation (degrees)')
    plt.title(f'Stability of Multi-Planet System using {method_name}')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f'stability_plot_{method_name}.png')
    plt.show()

# Main execution
def main():
    use_forest_ruth = True  # Set this to False to use leapfrog method

    method_name = "Forest-Ruth" if use_forest_ruth else "Leapfrog"
    planet_numbers, stability_measures = run_simulations(use_forest_ruth)
    plot_results(planet_numbers, stability_measures, method_name)

if __name__ == "__main__":
    main()
