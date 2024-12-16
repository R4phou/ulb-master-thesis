import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import Counter
import pydot
from graphviz import Source
from IPython.display import Image, display

# Function to calculate entropy
def entropy(labels):
    counts = Counter(labels)
    probabilities = [count / len(labels) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities)

# Function to calculate information gain
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = sum((len(data[data[feature] == value]) / len(data)) * 
                           entropy(data[data[feature] == value][target]) 
                           for value in values)
    return total_entropy - weighted_entropy

# ID3 algorithm
def id3(data, features, target):
    labels = data[target]
    
    # If all labels are the same, return that label
    if len(set(labels)) == 1:
        return labels.iloc[0]
    
    # If no features are left, return the most common label
    if len(features) == 0:
        return labels.mode()[0]
    
    # Select the feature with the highest information gain
    best_feature = max(features, key=lambda feature: information_gain(data, feature, target))
    
    # Create a tree with the best feature as the root
    tree = {best_feature: {}}
    
    # Remove the best feature from the list of features
    remaining_features = [f for f in features if f != best_feature]
    
    # Split the dataset on the best feature
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = id3(subset, remaining_features, target)
    
    return tree

# Sample dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Features and target
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
target = 'PlayTennis'

# Build the decision tree
decision_tree = id3(data, features, target)
print(decision_tree)


def plot_tree(tree, parent_name='', graph=None):
    if graph is None:
        graph = pydot.Dot(graph_type='graph')
    for node, branches in tree.items():
        if isinstance(branches, dict):
            node_name = f'{node}'
            graph.add_node(pydot.Node(node_name, label=node_name))
            if parent_name:
                graph.add_edge(pydot.Edge(parent_name, node_name))
            plot_tree(branches, node_name, graph)
        else:
            node_name = f'{parent_name}_{node}_{branches}'
            graph.add_node(pydot.Node(node_name, label=branches, shape='box'))
            if parent_name:
                graph.add_edge(pydot.Edge(parent_name, node_name))
    return graph


graph = plot_tree(decision_tree)
graph.write_png('decision_tree.png')
display(Image(filename='decision_tree.png'))

