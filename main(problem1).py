import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#-------------Data Preprocessing----------#
# Load the dataset
df = pd.read_csv('drug.csv')

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())
print("--------------------------------")

# Handle missing values
# For numerical columns, replace missing values with the mean
numeric_columns = ['Age', 'Na_to_K']
for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# For categorical columns, replace missing values with the most frequent value (mode)
categorical_columns = ['Sex', 'BP', 'Cholesterol']
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Verify that missing values have been handled
print("\nMissing values after handling:")
print(df.isnull().sum())
print("--------------------------------")

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['BP'] = label_encoder.fit_transform(df['BP'])
df['Cholesterol'] = label_encoder.fit_transform(df['Cholesterol'])

# Display the updated dataset
print("\nUpdated dataset:")
print(df.head())
print("--------------------------------")

# Assuming 'Drug' is your target variable
X = df.drop('Drug', axis=1)
y = df['Drug']

#--------------END Data Preprocessing------------------#

#-------------First experiment: Training and Testing with Fixed Train-Test Split Ratio:-------#

# Set the number of experiments
num_experiments = 5

# Initialize a list to store the accuracies and sizes of each experiment
results_fixed_split = []

# Repeat the experiment five times
for i in range(num_experiments):
    # Split the data into training and testing sets with a fixed ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # Create a Decision Tree model
    model_fixed_split = DecisionTreeClassifier(random_state=42)

    # Train the model on the training set
    model_fixed_split.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred_fixed_split = model_fixed_split.predict(X_test)

    # Calculate accuracy
    accuracy_fixed_split = accuracy_score(y_test, y_pred_fixed_split)

    # Get the size of the decision tree
    tree_size_fixed_split = model_fixed_split.tree_.node_count

    # Store the results
    results_fixed_split.append({
        'experiment': i + 1,
        'train_set_size': len(X_train),
        'test_set_size': len(X_test),
        'tree_size': tree_size_fixed_split,
        'accuracy': accuracy_fixed_split
    })

    # Print the results for each experiment
    print(f"\nExperiment {i + 1} (Fixed Split):")
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Decision Tree Size: {tree_size_fixed_split}")
    print(f"Decision Tree Accuracy: {accuracy_fixed_split:.4f}")
    print("--------------------------------")

# Sort the results based on accuracy in descending order
sorted_results_fixed_split = sorted(results_fixed_split, key=lambda x: x['accuracy'], reverse=True)

# Select the best-performing model (the one with the highest accuracy)
best_model_fixed_split = sorted_results_fixed_split[0]

# Print details of the best-performing model for fixed split
print("\nBest Performing Model (Fixed Split):")
print(f"Experiment {best_model_fixed_split['experiment']}:")
print(f"Train set size: {best_model_fixed_split['train_set_size']}, Test set size: {best_model_fixed_split['test_set_size']}")
print(f"Decision Tree Size: {best_model_fixed_split['tree_size']}")
print(f"Decision Tree Accuracy: {best_model_fixed_split['accuracy']:.4f}")
print("--------------------------------")
print("--------------------------------")

#-------------------------END First experiment ---------------------------#

#-------------Second experiment: Training and Testing with a Range of Train-Test Split Ratios:-------#

# Set the range of training set sizes
train_set_sizes = np.arange(0.3, 0.8, 0.1)

# Initialize lists to store statistics
train_set_sizes_report = []
mean_accuracies_report = []
max_accuracies_report = []
min_accuracies_report = []
mean_tree_sizes_report = []
max_tree_sizes_report = []
min_tree_sizes_report = []

# Initialize lists to store data for plots
accuracy_vs_size_data = {'Train Set Size': [], 'Mean Accuracy': []}
tree_size_vs_size_data = {'Train Set Size': [], 'Mean Tree Size': []}

# Initialize iteration counter
iteration_counter = 1

# Repeat the experiment for each training set size
for train_size in train_set_sizes:
    # Initialize lists to store statistics for each random seed
    accuracies = []
    tree_sizes = []
    print(f"Iteration {iteration_counter} for Train Set Size {train_size * 100}%:")
    # Repeat the experiment with five different random seeds
    for i in range(num_experiments):
        # Split the data into training and testing sets with a variable ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=i)

        # Create a Decision Tree model
        model_variable_split = DecisionTreeClassifier(random_state=i)

        # Train the model on the training set
        model_variable_split.fit(X_train, y_train)

        # Make predictions on the testing set
        y_pred_variable_split = model_variable_split.predict(X_test)

        # Calculate accuracy
        accuracy_variable_split = accuracy_score(y_test, y_pred_variable_split)

        # Get the size of the decision tree
        tree_size_variable_split = model_variable_split.tree_.node_count

        # Store accuracy and tree size for each seed
        accuracies.append(accuracy_variable_split)
        tree_sizes.append(tree_size_variable_split)

        # Print the results for each experiment
        print(f"\nRandom Seed: {i + 1}")
        print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        print(f"Decision Tree Size: {tree_size_variable_split}")
        print(f"Decision Tree Accuracy: {accuracy_variable_split:.4f}")

    # Calculate mean, max, min statistics
    mean_accuracy = np.mean(accuracies)
    max_accuracy = np.max(accuracies)
    min_accuracy = np.min(accuracies)
    mean_tree_size = np.mean(tree_sizes)
    max_tree_size = np.max(tree_sizes)
    min_tree_size = np.min(tree_sizes)

    # Append statistics to the report lists
    train_set_sizes_report.append(train_size)
    mean_accuracies_report.append(mean_accuracy)
    max_accuracies_report.append(max_accuracy)
    min_accuracies_report.append(min_accuracy)
    mean_tree_sizes_report.append(mean_tree_size)
    max_tree_sizes_report.append(max_tree_size)
    min_tree_sizes_report.append(min_tree_size)

    # Append data for plots
    accuracy_vs_size_data['Train Set Size'].append(train_size)
    accuracy_vs_size_data['Mean Accuracy'].append(mean_accuracy)
    tree_size_vs_size_data['Train Set Size'].append(train_size)
    tree_size_vs_size_data['Mean Tree Size'].append(mean_tree_size)

    # Increment the iteration counter
    iteration_counter += 1
    print("------------------------------------")

# Create a DataFrame to store the report
report_df = pd.DataFrame({
    'Train Set Size': train_set_sizes_report,
    'Mean Accuracy': mean_accuracies_report,
    'Max Accuracy': max_accuracies_report,
    'Min Accuracy': min_accuracies_report,
    'Mean Tree Size': mean_tree_sizes_report,
    'Max Tree Size': max_tree_sizes_report,
    'Min Tree Size': min_tree_sizes_report
})

# Display the report
print("\nExperiment Report:")
print(report_df)

# Save the report to a CSV file
report_df.to_csv('experiment_report.csv', index=False)

# Plot 1: Accuracy vs. Training Set Size
plt.figure(figsize=(10, 6))
plt.plot(accuracy_vs_size_data['Train Set Size'], accuracy_vs_size_data['Mean Accuracy'], marker='o')
plt.title('Accuracy vs. Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.show()

# Plot 2: Mean Tree Size vs. Training Set Size
plt.figure(figsize=(10, 6))
plt.plot(tree_size_vs_size_data['Train Set Size'], tree_size_vs_size_data['Mean Tree Size'], marker='o', color='orange')
plt.title('Mean Tree Size vs. Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Tree Size')
plt.grid(True)
plt.show()