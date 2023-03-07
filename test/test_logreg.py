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
import numpy as np
from sklearn.model_selection import train_test_split
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Implement tests for LogisticRegressor methods
def test_prediction():
	"""
	Check that predictions for LogisticRegressor are accurate
	"""
	np.random.seed(42)

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
	log_model.W = np.ones(num_feats + 1).flatten()
	log_model.train_model(X_train, y_train, X_val, y_val)

	sklearn_check = LogisticRegression().fit(X_val, y_val)
	#print("probs: ", sklearn_check.predict_proba(X_val))
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	assert np.allclose(log_model.make_prediction(X_val)[-5:], np.array([3.59265609e-01, 6.98316749e-01, 9.97294628e-01, 7.89724851e-01, 1.6522823e-01]))

	#assert npallclose(sklearn_check.predict_proba(X_val), log_model.make_prediction(X_val), atol=0.1)

	#mse = (np.square(sklearn_check.predict_proba(X_val) - log_model.make_prediction(X_val))).mean
	#assert mse > 0.5

def test_loss_function():
	"""
	Check that loss function for LogisticRegressor is computing loss correctly and compare to sklearn log_loss
	"""
	np.random.seed(42)

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
	log_model.W = np.ones(num_feats + 1).flatten()
	log_model.train_model(X_train, y_train, X_val, y_val)
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	assert log_loss(y_val, log_model.make_prediction(X_val)) == log_model.loss_function(y_val, log_model.make_prediction(X_val))

def test_gradient():
	"""
	Check that gradient function for LogisticRegressor is computing gradients correctly
	"""
	np.random.seed(42)

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
	log_model.W = np.ones(num_feats + 1).flatten()
	log_model.train_model(X_train, y_train, X_val, y_val)
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	assert np.allclose(log_model.calculate_gradient(y_val, X_val), np.array([-0.05655189, -0.06838798, -0.0583215, 0, 0, 0.07732318, 0.14970628]))

def test_training():
	"""
	Check that training function for LogisticRegressor is training data correctly by comparing to the loss history
	"""
	np.random.seed(42)

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
	log_model.W = np.ones(num_feats + 1).flatten()
	log_model.train_model(X_train, y_train, X_val, y_val)
	
	assert np.allclose(log_model.loss_hist_val[-5:], np.array([0.47525865970286346, 0.4752577720308588, 0.47525801529522227, 0.47525765772399065, 0.47525717698501724]))
