import numpy as np
from sklearn.svm import SVC

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

def main():
	X_train, Y_train, X_test, Y_test = read_data()

	# TODO: Implement problem 4a
	pass

if __name__ == '__main__':
	main()