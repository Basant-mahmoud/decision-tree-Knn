import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Data Preprocessing

# Separate features and targets
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data int training dataset and test data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Normalizing features using Min-Max Scaling
X_train_normalized = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test_normalized = (X_test - X_test.min()) / (X_test.max() - X_test.min())

# Converting to numpy arrays to deal with it with code
training_data = np.column_stack((X_train_normalized.values, y_train.values))
test_data = np.column_stack((X_test_normalized.values, y_test.values))

print("Training Data after Preprocessing:")
print(pd.DataFrame(training_data, columns=data.columns).head())

print("\nTest Data after Preprocessing:")
print(pd.DataFrame(test_data, columns=data.columns).head())


# Function to calculate the Euclidean Distance
def get_euclidean_distance(first, second):
    return np.sqrt(np.sum((first - second) ** 2))


# KNN Algorithm Logic
def knn_algorithm(training_data, test_instance, k):
    # Initialize array of distances ( each tuple in this array consists of distance and the corresponding training
    # instance.
    distances = []

    # Iterate all over the trainig data instance and calculate distance
    for i in range(len(training_data)):
        # select all row in training data except last element (target)
        dist = get_euclidean_distance(test_instance, training_data[i, :-1])

        # Each distance is appended to the distances list with the corresponding training instance.
        distances.append((training_data[i], dist))

    # Sort the distances list by distance and select the top k elements
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]

    # use dictionary where key is the targets (classes) and its count is the value
    class_counts = {}
    for neighbor in neighbors:
        # get the class label of the neighbor ( get the target ).
        # neighbor[0] refers to the neighbor's data ( that is the first thing that is appended in distances array,
        # and [-1] refers to the last element in this data, which is the class label ( target ).
        response = neighbor[0][-1]
        if response in class_counts:
            class_counts[response] += 1
        else:
            class_counts[response] = 1

    # Sort the class_counts dictionary by votes and return the class with the most votes
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    # Check for a tie
    if len(sorted_counts) > 1 and sorted_counts[0][1] == sorted_counts[1][1]:
        # Apply distance-weighted voting in case of a tie
        # First we must clear our dictionary
        class_counts.clear()
        for neighbor in neighbors:
            response = neighbor[0][-1]  # Get the class label of the neighbor whether it's 0 or 1

            # weight is inversily proportional to distance so lower distance have high weights and vice versa
            if neighbor[1] != 0:
                weight = 1 / neighbor[1]
            # if it 0 then it must be a very very big number
            else:
                weight = float('inf')

            # update the value of the target counts
            if response in class_counts:
                class_counts[response] += weight
            else:
                class_counts[response] = weight

        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    return sorted_counts[0][0]


# function which tests our knn algorithm
def evaluate_knn(k_values, training_data, test_data):
    # array to store all the accuaracies that used for calc avg later
    accuracies = []

    # loop over the k values
    for k in k_values:
        # see how many correct predictions our model make
        correct_predictions = 0

        # for each test instance in test data .. see whether it predict correctly or not
        for test_instance in test_data:
            predicted_class = knn_algorithm(training_data, test_instance[:-1], k)
            if predicted_class == test_instance[-1]:
                correct_predictions += 1

        accuracy = correct_predictions / len(test_data)
        accuracies.append((k, accuracy))

        print(f"k value: {k}")
        print(f"Number of correctly classified instances: {correct_predictions}")
        print(f"Total number of instances: {len(test_data)}")
        print(f"Accuracy: {accuracy * 100:.2f}%")

    # Output the average accuracy across all iterations
    average_accuracy = sum(accuracy for _, accuracy in accuracies) / len(accuracies)
    print(f"Average accuracy across all iterations: {average_accuracy * 100:.2f}%")

    return accuracies


k_values = [2, 3, 4, 5, 6]

evaluate_knn(k_values, training_data, test_data)
