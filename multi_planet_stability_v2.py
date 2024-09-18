import numpy as np
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
G = 6.67430e-11  # Gravitational constant
M_SUN = 1.989e30  # Mass of the Sun in kg
M_EARTH = 5.972e24  # Mass of the Earth in kg
AU = 1.496e11  # 1 AU in meters
YEAR = 365.25 * 24 * 3600  # 1 year in seconds

# CUDA kernel for calculating accelerations
cuda_code = """
__global__ void calculate_acceleration(double *pos, double *acc, double *masses, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double ax = 0.0, ay = 0.0;
        double xi = pos[i*2], yi = pos[i*2+1];
        double mi = masses[i];
        
        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                double dx = pos[j*2] - xi;
                double dy = pos[j*2+1] - yi;
                double dist_sqr = dx*dx + dy*dy;
                double inv_dist_cube = rsqrt(dist_sqr * dist_sqr * dist_sqr);
                double mj = masses[j];
                double fac = mj * inv_dist_cube;
                ax += fac * dx;
                ay += fac * dy;
            }
        }
        acc[i*2] = (ax * 6.67430e-11) / mi;
        acc[i*2+1] = (ay * 6.67430e-11) / mi;
    }
}
"""

# Compile CUDA kernel
mod = SourceModule(cuda_code)
calculate_acceleration_gpu = mod.get_function("calculate_acceleration")

def initialize_system(num_planets, perturbation_angle):
    positions = np.zeros((num_planets + 1, 2))
    velocities = np.zeros((num_planets + 1, 2))
    masses = np.array([M_SUN] + [M_EARTH] * num_planets)
    
    angles = np.linspace(0, 2*np.pi, num_planets, endpoint=False)
    positions[1:, 0] = AU * np.cos(angles)
    positions[1:, 1] = AU * np.sin(angles)
    
    v = np.sqrt(G * M_SUN / AU)
    velocities[1:, 0] = -v * np.sin(angles)
    velocities[1:, 1] = v * np.cos(angles)
    
    perturbation = perturbation_angle * np.pi / 180
    positions[1] = np.array([AU * np.cos(perturbation), AU * np.sin(perturbation)])
    velocities[1] = np.array([-v * np.sin(perturbation), v * np.cos(perturbation)])
    
    return positions, velocities, masses

def leapfrog_step_gpu(pos_gpu, vel_gpu, acc_gpu, masses_gpu, dt, num_bodies):
    block_size = 256
    grid_size = (num_bodies + block_size - 1) // block_size

    vel_gpu += 0.5 * acc_gpu * dt
    pos_gpu += vel_gpu * dt
    calculate_acceleration_gpu(
        pos_gpu, acc_gpu, masses_gpu, np.int32(num_bodies),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    vel_gpu += 0.5 * acc_gpu * dt

def calculate_total_energy(pos, vel, masses):
    kinetic = 0.5 * np.sum(masses * np.sum(vel**2, axis=1))
    potential = 0
    for i in range(len(masses)):
        for j in range(i+1, len(masses)):
            r = np.linalg.norm(pos[i] - pos[j])
            potential -= G * masses[i] * masses[j] / r
    return kinetic + potential

def calculate_angular_momentum(pos, vel, masses):
    return np.sum(masses[:, np.newaxis] * np.cross(pos, vel), axis=0)

def orbital_period(radius, central_mass):
    return 2 * np.pi * np.sqrt(radius**3 / (G * central_mass))

def expected_angle(initial_angle, time, orbital_period):
    return (initial_angle + (2 * np.pi * time / orbital_period)) % (2 * np.pi)

def calculate_angular_deviation(initial_angles, current_angles, times, periods):
    expected_angles = expected_angle(initial_angles, times, periods)
    deviation = np.abs(current_angles - expected_angles)
    deviation = np.minimum(deviation, 2*np.pi - deviation)
    return np.max(deviation)

def adaptive_time_step(positions, base_dt):
    min_dist = np.min([np.linalg.norm(pos1 - pos2) for i, pos1 in enumerate(positions) for pos2 in positions[i+1:]])
    return min(base_dt, 0.01 * min_dist / np.sqrt(G * M_SUN))

def simulate_gpu(num_planets, years, base_dt, perturbation_angle):
    positions, velocities, masses = initialize_system(num_planets, perturbation_angle)
    num_bodies = num_planets + 1
    max_deviation = 0

    radii = np.linalg.norm(positions[1:], axis=1)
    periods = orbital_period(radii, masses[0])

    pos_gpu = gpuarray.to_gpu(positions.astype(np.float64))
    vel_gpu = gpuarray.to_gpu(velocities.astype(np.float64))
    acc_gpu = gpuarray.zeros_like(pos_gpu)
    masses_gpu = gpuarray.to_gpu(masses.astype(np.float64))

    initial_angles = np.arctan2(positions[1:, 1], positions[1:, 0])
    initial_energy = calculate_total_energy(positions, velocities, masses)
    initial_angular_momentum = calculate_angular_momentum(positions, velocities, masses)

    total_time = 0
    target_time = years * YEAR
    step = 0
    energy_deviation = []
    angular_momentum_deviation = []

    while total_time < target_time:
        positions = pos_gpu.get()
        dt = adaptive_time_step(positions, base_dt)
        dt = min(dt, target_time - total_time)

        leapfrog_step_gpu(pos_gpu, vel_gpu, acc_gpu, masses_gpu, dt, num_bodies)

        total_time += dt
        step += 1

        if step % 100 == 0:
            positions = pos_gpu.get()
            velocities = vel_gpu.get()
            current_angles = np.arctan2(positions[1:, 1], positions[1:, 0])
            deviation = calculate_angular_deviation(initial_angles, current_angles, total_time, periods)
            max_deviation = max(max_deviation, deviation)

            current_energy = calculate_total_energy(positions, velocities, masses)
            energy_dev = (current_energy - initial_energy) / initial_energy
            energy_deviation.append(energy_dev)

            current_angular_momentum = calculate_angular_momentum(positions, velocities, masses)
            am_dev = np.linalg.norm(current_angular_momentum - initial_angular_momentum) / np.linalg.norm(initial_angular_momentum)
            angular_momentum_deviation.append(am_dev)

            #logging.info(f"Time: {total_time/YEAR:.2f} years, Max Deviation: {max_deviation * 180/np.pi:.2f} degrees, "
            #             f"Energy Dev: {energy_dev:.2e}, Angular Momentum Dev: {am_dev:.2e}")

    return max_deviation * 180 / np.pi, energy_deviation, angular_momentum_deviation

def run_simulations(num_bodies, num_years):
    planet_numbers = range(2, num_bodies)
    stability_measures = []
    energy_deviations = []
    angular_momentum_deviations = []

    for N in planet_numbers:
        logging.info(f"Simulating {N} planets...")
        start = time.time()
        max_deviation, energy_dev, am_dev = simulate_gpu(N, years=num_years, base_dt=1*24*3600, perturbation_angle=0.1)
        #print(f"max_deviation={max_deviation} for {N} planets in {time.time() - start} seconds.")
        stability_measures.append(max_deviation)
        energy_deviations.append(np.mean(np.abs(energy_dev)))
        angular_momentum_deviations.append(np.mean(np.abs(am_dev)))

    return planet_numbers, stability_measures, energy_deviations, angular_momentum_deviations

def plot_results(planet_numbers, stability_measures, energy_deviations, angular_momentum_deviations):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    ax1.plot(planet_numbers, stability_measures, '-o')
    ax1.set_xlabel('Number of Planets')
    ax1.set_ylabel('Maximum Angular Deviation (degrees)')
    ax1.set_title('Stability of Multi-Planet System')
    ax1.grid(True)
    ax1.set_yscale('log')

    ax2.plot(planet_numbers, energy_deviations, '-o')
    ax2.set_xlabel('Number of Planets')
    ax2.set_ylabel('Mean Absolute Energy Deviation')
    ax2.set_title('Energy Conservation')
    ax2.grid(True)
    ax2.set_yscale('log')

    ax3.plot(planet_numbers, angular_momentum_deviations, '-o')
    ax3.set_xlabel('Number of Planets')
    ax3.set_ylabel('Mean Absolute Angular Momentum Deviation')
    ax3.set_title('Angular Momentum Conservation')
    ax3.grid(True)
    ax3.set_yscale('log')

    plt.tight_layout()
    plt.savefig('multi_planet_stability_analysis.png')
    plt.show()

def main():
    # Main execution
    num_bodies = int(input("how many bodies? "))
    num_years = int(input("how many years?" ))
    print("BEGIN SIMULATION")
    planet_numbers, stability_measures, energy_deviations, angular_momentum_deviations = run_simulations(num_bodies, num_years)
    print("END SIMULATION")
    plot_results(planet_numbers, stability_measures, energy_deviations, angular_momentum_deviations)

if __name__ == "__main__":
    main()
