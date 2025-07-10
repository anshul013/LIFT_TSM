import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_samples = 1000
time = np.arange(num_samples)
noise_level = 0.5

# Generate the first channel (random walk)
channel_1 = np.cumsum(np.random.normal(scale=noise_level, size=num_samples))

# Intentional error: Using an undefined variable
# Let's say we mistakenly use 'channel_2' instead of 'channels[1]'
plt.plot(time, channel_2, label='Channel 2', alpha=0.8)

# Create phase shifts for the other channels
phase_shifts = [0, 30, 60, 90]  # in degrees

# Generate the other channels with phase shifts
channels = [channel_1]
for shift in phase_shifts:
    radians_shift = np.radians(shift)
    channel = np.sin(time * np.pi / 180 + radians_shift) + np.random.normal(scale=noise_level, size=num_samples)
    channels.append(channel)

# Save the data in a .csv file inside the /dataset directory
print()

# # Plot the synthetic time series data
# plt.figure(figsize=(10, 6))
# plt.plot(time, channel_1, label='Channel 1 (Random Walk)', linewidth=2)
# plt.plot(time, channels[1], label='Channel 2', alpha=0.8)
# plt.plot(time, channels[2], label='Channel 3', alpha=0.8)
# plt.plot(time, channels[3], label='Channel 4', alpha=0.8)
# plt.plot(time, channels[4], label='Channel 5', alpha=0.8)

# plt.title('Synthetic Time Series Data with Correlated Channels and Phase Shifts')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
