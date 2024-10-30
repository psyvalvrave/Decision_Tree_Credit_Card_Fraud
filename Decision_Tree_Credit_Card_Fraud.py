# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:23:47 2024

@author: Zhecheng Li
"""

import pandas as pd 
import numpy as np
from sklearn import tree
from datetime import datetime
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None
        
    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])
    
    def information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self.entropy(y)

        # create children
        left_index, right_index = self.split(X_column, threshold)

        if len(left_index) == 0 or len(right_index) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_index), len(right_index)
        e_l, e_r = self.entropy(y[left_index]), self.entropy(y[right_index])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the Information Gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self.create_tree(X, y)

    def create_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        feat_index = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_threshold = self.best_split(X, y, feat_index)

        # create child nodes after split 
        left_index, right_index = self.split(X[:, best_feature], best_threshold)
        left = self.create_tree(X[left_index, :], y[left_index], depth+1)
        right = self.create_tree(X[right_index, :], y[right_index], depth+1)
        return Node(best_feature, best_threshold, left, right)
    
    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)


    def best_split(self, X, y, feat_index):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_index:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for i in thresholds:
                # calculate the information current_gain
                current_gain = self.information_gain(y, X_column, i)

                if current_gain > best_gain:
                    best_gain = current_gain
                    split_idx = feat_idx
                    split_threshold = i

        return split_idx, split_threshold
    

    def split(self, X_column, split_threshold):
        left_index = np.argwhere(X_column <= split_threshold).flatten()
        right_index = np.argwhere(X_column > split_threshold).flatten()
        return left_index, right_index


    def most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])



df = pd.read_csv("fraud test.csv")

df = df.drop(
    ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 'street', 'city', 'state', 'zip',
     'lat', 'long', 'city_pop', 'job', 'trans_num', 'unix_time', 'merch_lat', 'merch_long'], axis=1) #Drop Unncessary features.

df['dob'] = pd.to_datetime(df['dob'], dayfirst=True)
current_date = datetime.now()
df['Age'] = (current_date - df['dob']).dt.days // 365 # Drop dob feature and convert it to Age
df = df.drop('dob', axis=1)

plt.figure(figsize=(10, 8))
sns.countplot(x='category', data=df)
plt.xlabel('Category')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.show()

#should we keep?
# Calculate the total transaction amount by gender
total_amount_by_gender = df.groupby('gender')['amt'].sum()

# Plot
plt.figure(figsize=(8, 6))
plt.pie(total_amount_by_gender, labels=total_amount_by_gender.index, autopct='%1.1f%%', startangle=140)
plt.title('Total Transaction Amounts by Gender')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='amt', data=df, hue='gender', style='gender', alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Transaction Amount')
plt.show()

import random
if random.random() > 0.5:
    df = df.head(100000)  
else:
    df = df.tail(100000)  

df_encoded = pd.get_dummies(df, columns=['category', 'gender'], dtype=int, drop_first=True) # Convert Category & Gender from Object to Int

df = df_encoded
X = df.drop('is_fraud', axis=1) #Input 
X = X.to_numpy()
Y = df['is_fraud'] #Output
Y = Y.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle= True)

smote = SMOTE(sampling_strategy='auto') #To Balance the data as output is imbalance in the dataset.
x_train_smoted, y_train_smoted = smote.fit_resample(x_train, y_train)

testModel = DecisionTree(max_depth=10)
testModel.fit(x_train_smoted, y_train_smoted)
Y_pred = testModel.predict(x_test) 

model = tree.DecisionTreeClassifier(random_state=42) 
model = model.fit(x_train_smoted, y_train_smoted)
Ypred = model.predict(x_test)

accuracy = accuracy_score(y_test, Y_pred)
accuracy_sklearn = accuracy_score(y_test, Ypred)
print("accuracy for self-built tree: ", accuracy)
print("accuracy for sklearn built-in tree: ", accuracy_sklearn)

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
y_train_pred = testModel.predict(x_train_smoted)

print("Training Accuracy:", accuracy_score(y_train_smoted, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, Y_pred))
print("Training Precision:", precision_score(y_train_smoted, y_train_pred, average='binary'))
print("Testing Precision:", precision_score(y_test, Y_pred, average='binary'))
print("Training Recall:", recall_score(y_train_smoted, y_train_pred, average='binary'))
print("Testing Recall:", recall_score(y_test, Y_pred, average='binary'))

classification_rep = classification_report(y_test, Y_pred)
print("Classification report:\n", classification_rep)