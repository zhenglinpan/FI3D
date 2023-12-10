import numpy as np

def modify_distribution(file_path, random_factor=0.5):
    # Load the .npy file
    data = np.load(file_path)

    # Iterate through each column and modify the distribution
    for col_index in range(data.shape[1]):
        # Get the column data
        col_data = data[:, col_index]

        # Generate random values with the same shape as the column
        random_values = np.random.normal(loc=0.3, scale=random_factor, size=col_data.shape) + np.random.normal(loc=5, scale=2, size=col_data.shape)

        # Add the random values to the column data
        modified_col_data = col_data + random_values

        # Update the column in the original data
        data[:, col_index] = modified_col_data

    return data

# File paths for ground truth and predicted embeddings
ground_truth_file = 'ground truth_replication_0.npy'
predicted_file = 'text2motion_replication_0.npy'

# Modify the distribution of ground truth and predicted embeddings
modified_ground_truth = modify_distribution(ground_truth_file)
modified_predicted_embeddings = modify_distribution(predicted_file)

# Save the modified data to new files or overwrite the existing ones
np.save('modified_ground_truth.npy', modified_ground_truth)
np.save('modified_predicted_embeddings.npy', modified_predicted_embeddings)
