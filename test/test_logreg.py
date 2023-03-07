"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

def test_prediction():

	X_train, X_val, y_train, y_val = utils.loadDataset(
	    features=[
	        'Penicillin V Potassium 500 MG',
	        'Computed tomography of chest and abdomen',
	        'Plain chest X-ray (procedure)',
	        'Low Density Lipoprotein Cholesterol',
	        'Creatinine',
	        'AGE_DIAGNOSIS'
	    ],
	    split_percent=0.8,
	    split_seed=42
	)

	num_feats = 6
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	LogisticRegression()
	log_model.train_model(X_train, y_train, X_val, y_val)

	logistic_check = LogisticRegression().fit(X_val, y_val)
	#assert np.allclose(logistic_check.predict(X_val)

	#[3.59265609e-01, 6.98316749e-01, 9.97294628e-01, 7.89724851e-01, 1.65228229e-01]
	print("func: ", X_val.shape)


def test_loss_function():
	pass

def test_gradient():
	pass

def test_training():
	pass