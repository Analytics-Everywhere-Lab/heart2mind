import numpy as np
import pickle
from utils import load_data


def extract_ground_truths(y, patient_ids):
    unique_patient_ids = np.unique(patient_ids)
    ground_truths = []

    for patient_id in unique_patient_ids:
        patient_indices = np.where(patient_ids == patient_id)[0]
        patient_label = y[patient_indices][0]
        ground_truths.append(patient_label)

    return ground_truths


def save_ground_truths(ground_truths, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(ground_truths, f)


def main():
    # Load data
    X, y, patient_ids = load_data('data.pkl')

    # Extract ground truths
    ground_truths = extract_ground_truths(y, patient_ids)

    # Print the list of ground truths
    print("Ground Truths:", ground_truths)

    # Save ground truths
    save_ground_truths(ground_truths, 'ground_truths.pkl')

    print("Ground truths extracted and saved successfully.")


if __name__ == "__main__":
    main()
