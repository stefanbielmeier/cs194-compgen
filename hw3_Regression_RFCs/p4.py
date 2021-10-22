import numpy as np
from sklearn.svm import SVC
import time

"""
Problem 4 – Training a Support Vector Machine and Cross-Validation
"""


DATA_CSV = "pulsar_data.csv"

def read_data():
	# DO NOT modify this function

	data = np.genfromtxt(DATA_CSV, skip_header=1, delimiter=',')
	X, Y = data[:, : 8], data[:, 8]
	X = np.nan_to_num(X)

	permutation = np.random.permutation(X.shape[0])
	train_idxs = permutation[: 1_000]
	test_idxs = permutation[1_000: 2_000]

	X_train, Y_train = X[train_idxs], Y[train_idxs]
	X_test, Y_test = X[test_idxs], Y[test_idxs]
	return X_train, Y_train, X_test, Y_test

def train_linear(x, y, C = 1):
	classifier = SVC(C=C, kernel='linear')
	train = classifier.fit(x, y)
	return train

def train_poly2(x, y, C = 1):
	classifier = SVC(C=C, kernel='poly', degree=2)
	train = classifier.fit(x, y)
	return train

def train_poly5(x, y, C = 1):
	classifier = SVC(C=C, kernel='poly', degree=5)
	train = classifier.fit(x, y)
	return train

def train_rbf(x, y, C = 1):
	classifier = SVC(C=C, kernel='rbf')
	train = classifier.fit(x, y)
	return train

def train_svm(training_set_X, training_set_Y, classifier='linear', C=1):
	if classifier == "linear":
		return train_linear(training_set_X, training_set_Y, C=C)		
	elif classifier == "poly2":		
		return train_poly2(training_set_X, training_set_Y, C=C)
	elif classifier == "poly5":		
		return train_poly5(training_set_X, training_set_Y, C=C)
	elif classifier == "rbf":
		return train_rbf(training_set_X, training_set_Y, C=C)
	else:
		print('error. Pick a right model')
		return False


def cross_validation(X_train, Y_train, k = 5, classifier='linear', C = 1):
	"""
	Arguments: a randomized training dataset of X_values and Y_values that correspond to each other, and k which is the k-folds of the crossvalidation
	Returns: mean accuracy over all k-folds after training
	"""
	sum_acc = 0
	k_size = int(X_train.shape[0] / k)

	#for k times 
	for i in range(k):
		#assemble training & validation set combinations of X and Y, leaving out 1/kth of the set for validation
		validation_set_X = X_train[i*k_size : (i+1)*k_size, :]
		validation_set_Y = Y_train[i*k_size : (i+1)*k_size]
		
		training_set_X = np.delete(X_train, np.s_[i*k_size:(i+1)*k_size], axis=0) 
		training_set_Y = np.delete(Y_train, np.s_[i*k_size:(i+1)*k_size], axis=0)
	
		#train model with specified C
		model = train_svm(training_set_X, training_set_Y, classifier, C=C)
		predictions = model.predict(validation_set_X)
	
		#calculate accuracy of classifier on respective validation, add to mean_acc
		
		correct_instances = np.equal(predictions, validation_set_Y)
		
		sum_acc += np.size(correct_instances[correct_instances == True]) / k_size

	#return mean accuracy across the k folds
	mean_acc = sum_acc / k
	return mean_acc


def main():
	X_train, Y_train, X_test, Y_test = read_data()

	start = time.time()
	for i in range(5):
		print("acc for poly2 for C={}".format(0.1*(10**i)), cross_validation(X_train, Y_train, k=5, classifier='linear', C=0.1*(10**i)))	
	end = time.time()
	print("cross-validation took {}".format(round(end-start, 1)), "secs")
	
	start = time.time()
	for i in range(5):
		print("acc for poly5 for C={}".format(0.1*(10**i)), cross_validation(X_train, Y_train, k=5, classifier='poly5', C=0.1*(10**i)))
	end = time.time()
	print("cross-validation took {}".format(round(end-start, 1)), "secs")

	for i in range(5):
		start = time.time()
		print("acc for rbf for C={}".format(0.1*(10**i)), cross_validation(X_train, Y_train, k=5, classifier='rbf', C=0.1*(10**i)))	
	end = time.time()
	print("cross-validation took {}".format(round(end-start, 1)), "secs")

	start = time.time()
	for i in range(5):
		print("acc for linear for C={}".format(0.1*(10**i)), cross_validation(X_train, Y_train, k=5, classifier='linear', C=0.1*(10**i)))
	end = time.time()
	print("cross-validation took {}".format(round(end-start, 1)), "secs")
	
	#train best classifiers
	linear_classifier = train_linear(X_train, Y_train, C = 1)
	poly2_classifier = train_poly2(X_train, Y_train, C = 1)
	poly4_classifier = train_poly5(X_train, Y_train, C = 1)
	rbf_classifier = train_rbf(X_train, Y_train, C = 1)
	mean_acc = cross_validation(X_train, Y_train)

	#report test acc of best classifiers

	pass

if __name__ == '__main__':
	main()