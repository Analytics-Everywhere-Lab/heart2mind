import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compute_hrv_metrics(data, start_time, end_time):
    # Filter the data within the given timestamp range
    filtered_data = data[(data['Phone timestamp'] >= start_time) & (data['Phone timestamp'] <= end_time)]

    # Extract the RR intervals in milliseconds
    rr_intervals = filtered_data['RR-interval [ms]'].values

    # Calculate SDNN
    sdnn = np.std(rr_intervals)

    # Calculate RMSSD
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))

    # Calculate pNN50 (percentage of NN intervals greater than 50 ms)
    nn_diff = np.abs(np.diff(rr_intervals))
    pnn50 = np.sum(nn_diff > 50) / len(nn_diff) * 100 if len(nn_diff) > 0 else np.nan

    return {
        'SDNN': sdnn,
        'RMSSD': rmssd,
        'pNN50': pnn50
    }


def plot_and_compute_hrv_by_step(data, start_step, end_step):
    # Extract the relevant portion of the data based on step indices
    filtered_data = data.iloc[start_step:end_step]

    # Get the start and end times for highlighting
    start_time = filtered_data['Phone timestamp'].iloc[0]
    end_time = filtered_data['Phone timestamp'].iloc[-1]

    # Plot the full time series data with highlighted range
    plt.figure(figsize=(12, 6))
    plt.plot(data['Phone timestamp'], data['RR-interval [ms]'], linewidth=0.5, color='blue', label='RR-Interval')

    # Highlight the selected range
    plt.axvspan(start_time, end_time, color='yellow', alpha=0.3,
                label=f'Selected Range (Steps {start_step} to {end_step})')

    plt.xlabel('Timestamp', fontsize=14, weight='bold')
    plt.ylabel('RR-Interval (ms)', fontsize=14, weight='bold')
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.title('ECG Data with Selected Range Highlighted by Steps', fontsize=16, weight='bold')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('treatment_with_range_by_steps.pdf')
    plt.show()

    # Compute HRV metrics for the selected range
    hrv_metrics = compute_hrv_metrics(filtered_data, start_time, end_time)

    return hrv_metrics


if __name__ == '__main__':
    # Load and parse the CSV file
    file_path = 'HRV_anonymized_data/control_30.csv'
    data = pd.read_csv(file_path, sep=';')

    # Convert timestamp to datetime
    data['Phone timestamp'] = pd.to_datetime(data['Phone timestamp'])

    start_step = 7300
    end_step = 8000
    hrv_metrics_by_step = plot_and_compute_hrv_by_step(data, start_step, end_step)
    print(hrv_metrics_by_step)