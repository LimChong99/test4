# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 04:46:20 2025

@author: hongf
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris 

# Load the dataset
def load_data():
    data = load_iris(as_frame=True)  
    df = data['frame']
    df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in df.columns]
    df['target'] = data.target
    return df

# Preprocess the data
def preprocess(df):
    features = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    X = features
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train and evaluate the model
def train_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    train_model(X_train, X_test, y_train, y_test)
