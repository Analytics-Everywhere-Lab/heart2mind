import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and parse the CSV file
file_path = 'HRV_anonymized_data/treatment_4.csv'
data = pd.read_csv(file_path, sep=';')

# Convert timestamp to datetime
data['Phone timestamp'] = pd.to_datetime(data['Phone timestamp'])

# # Select only the first 20 data points
# data = data.iloc[:20]

# # Add Gaussian noise
# noise_std = 0.5  # Adjust standard deviation as needed
# data['RR-interval Noisy'] = data['RR-interval [ms]'] + np.random.normal(0, noise_std, len(data))

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = 'cm'
# Plot the original and noisy signals
plt.figure(figsize=(11, 6))
plt.plot(data['Phone timestamp'], data['RR-interval [ms]'], linewidth=0.25, color='darkblue')
# plt.plot(data['Phone timestamp'], data['RR-interval Noisy'], linewidth=1, color='blue', marker='s', alpha=0.6, label='Noisy')

# # Hide all ticks and labels
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
# plt.tick_params(
#     axis='y',          # changes apply to the y-axis
#     which='both',      # both major and minor ticks are affected
#     left=False,        # ticks along the left edge are off
#     right=False,       # ticks along the right edge are off
#     labelleft=False)   # labels along the left edge are off

plt.xlabel('Timestamp', fontsize=25, weight='bold')
# plt.ylabel('RR-Interval (ms)', fontsize=25, weight='bold')

# Customize x-axis labels
plt.xticks(fontsize=20, rotation=45)
plt.yticks(fontsize=20)

# plt.legend(fontsize=30)
plt.tight_layout()
plt.grid(True)

# Save the figure
plt.savefig('treatment_sample.pdf')
plt.show()
