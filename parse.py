import numpy as np
import csv

def _parse_features():
    # Parses and returns a dictionary with the features and their labels
    features = {}

    with open('attributelegend.txt') as f:
        # Skip over the class descriptions
        f.readline()
        # Iterate over the rest of the descriptions
        for line in f.readlines():
            # Skip empty lines
            if line == '\n':
                continue
            # The name of the feature
            primary_feature = line.split(':')[0]
            # Store the labels in a list
            features[primary_feature] = []
            for subfeature in line.split(':')[1].split(','):
                features[primary_feature].append(subfeature.split('=')[1].strip())
    # Return the dictionary
    return features

def parse_data():
    # Parses the csv file into the input and target values
    x = []
    y = []

    features = _parse_features()

    with open('mushrooms.csv') as f:
        # Skip over the headers
        f.readline()
        reader = csv.reader(f)
        # Iterate over the data
        for row in reader:
            # Save the target value as 1 if it's poisonous and 0 if it's edible
            y.append(int(row[0] == 'p'))
            # Save the rest of the features using one-hot encoding
            x.append([])
            for i, feature in enumerate(features):
                for label in features[feature]:
                    # TODO try skipping the missing label
                    if label == row[i + 1]:
                        x[-1].append(1)
                    else:
                        x[-1].append(0)
    return np.asarray(x), np.asarray(y)
