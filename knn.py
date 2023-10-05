
# -------------------------------------------------------------------------
# AUTHOR: Vu Nguyen
# FILENAME: knn.py
# SPECIFICATION: File used to compare the error rate of KNN against other ML algorithms
# FOR: CS 4210- Assignment #2
# TIME SPENT: 5 hrs
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays


import csv
from sklearn.neighbors import KNeighborsClassifier

data_entries = []
class_mapping = {"-": 1, "+": 2}
errors_count = 0

# Load the data from the CSV file
with open('binary_points.csv', 'r') as file:
    reader = csv.reader(file)
    for idx, line in enumerate(reader):
        if idx > 0:  # Omitting the header
            data_entries.append(line)

# Process each data entry for cross-validation
for test_entry in data_entries:

    # Separate training data and their corresponding classes
    training_data = [list(map(float, entry[:-1])) for entry in data_entries if entry != test_entry]
    training_labels = [float(class_mapping[entry[-1]]) for entry in data_entries if entry != test_entry]

    # Prepare the test data
    test_data = list(map(float, test_entry[:-1]))
    test_label = float(class_mapping[test_entry[-1]])

    # Initialize and train the KNN classifier
    classifier = KNeighborsClassifier(n_neighbors=1, p=2)
    classifier.fit(training_data, training_labels)

    # Predict the class of the test entry
    predicted_class = classifier.predict([test_data])[0]

    # Compare the prediction to the actual class
    if test_label != predicted_class:
        errors_count += 1

# Display the error rate
print(f"Error Rate: {errors_count / len(data_entries):.2f}")
