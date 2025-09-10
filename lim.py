# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 12:30:49 2025

@author: limch
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# --- Program Functions --- #

def load_data():
    """Loads the Iris dataset and cleans the column names."""
    data = load_iris(as_frame=True)
    df = data['frame']
    # Clean column names to use underscores instead of spaces
    df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in df.columns]
    return df

def preprocess(df):
    """Takes a clean DataFrame and prepares it for model training."""
    features = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    X = features
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_model(X_train, X_test, y_train, y_test):
    """Trains a model and returns its accuracy."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

# --- Unit Test Suite --- #

class TestProgram(unittest.TestCase):

    def setUp(self):
        """Load a clean dataset once before each test."""
        self.df = load_data()

    def test_data_loading(self):
        """Test that the loaded DataFrame has the right shape and target column."""
        self.assertIn('target', self.df.columns)
        self.assertEqual(self.df.shape[0], 150) # 150 rows in Iris dataset

    def test_column_names_are_clean(self):
        """Ensure the column names have been cleaned correctly by load_data."""
        expected_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
        self.assertListEqual(list(self.df.columns), expected_cols)

    def test_preprocess_output(self):
        """Ensure preprocess returns four datasets of the correct type and shape."""
        X_train, X_test, y_train, y_test = preprocess(self.df)
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(y_train, pd.Series)
        self.assertEqual(X_train.shape[0], 120) # 80% of 150
        self.assertEqual(X_test.shape[0], 30)   # 20% of 150

    def test_model_training_accuracy(self):
        """Test that the model trains and achieves a reasonable accuracy."""
        X_train, X_test, y_train, y_test = preprocess(self.df)
        accuracy = train_model(X_train, X_test, y_train, y_test)
        # Check that accuracy is plausible (e.g., better than 80%)
        self.assertGreater(accuracy, 0.8)

if __name__ == '__main__':
    unittest.main()






