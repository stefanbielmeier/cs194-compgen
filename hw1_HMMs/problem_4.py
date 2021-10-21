import math
import numpy as np
from numpy.core.fromnumeric import argmax 
from scipy import stats

### BEGIN: HMM model parameters

states_map = {
	0: "Nick",
	1: "Coach", 
	2: "Winston"
}

means = { # mean of each friend's driving speed (modeled by a Gaussian)
	0: 60.0,
	1: 65.0,
	2: 63.0
}

variances = { # variance of each friend's driving speed (modeled by a Gaussian)
	0: 5.0,
	1: 6.0,
	2: 8.0
}

# transition_matrix[row][col] = P(pi_{i} = col | pi_{i - 1} = row)
transition_matrix = np.asarray([ 
	[0.90, 0.05, 0.05],
	[0.10, 0.80, 0.10],
	[0.15, 0.15, 0.70]
])

### END: HMM model parameters

def read_data():
	# Helper function to read data.
	# DO NOT modify this function.
	
	num_samples = 10
	observation_length = 100

	observations_matrix = np.zeros((num_samples, observation_length), dtype=np.float32)
	most_probable_states = np.zeros((num_samples, observation_length), dtype=np.int32)
	observation_probs = np.zeros((num_samples), dtype=np.float64)

	with open("problem_4_data.txt") as f:
		for i in range(num_samples):
			observations = f.readline().strip().split(',')
			viterbi_output = f.readline().strip().split(',')
			forward_output = float(f.readline().strip())
			f.readline() # empty line

			observations_matrix[i] = np.asarray(list(map(float, observations)))
			most_probable_states[i] = np.asarray(list(map(int, viterbi_output)))
			observation_probs[i] = float(forward_output)

	return observations_matrix, most_probable_states, observation_probs

def run_viterbi(observations):
	# observations: a numpy array of driving speeds
	# returns: numpy array of states with the same length as observations
	
	# PI = array of intiial state probabilities: here: 1/3, everyone is equally likely.
	large_pi = np.array([1/3, 1/3, 1/3])

	# A = transition_matrix of format K * K
	
	# B = emission_matrix of format K * N where row(state) -> col(observation)
	# # for each possible observation in the observation space, calculate the P(X) 3 times for each state, and put it in matrix [state][observation]
	emission_matrix = np.zeros((3, len(observations)))

	for index in range(len(observations)):
		emission_matrix[0, index] = stats.norm.pdf(observations[index], means[0], math.sqrt(variances[0]))
		emission_matrix[1, index] = stats.norm.pdf(observations[index], means[1], math.sqrt(variances[1]))
		emission_matrix[2, index] = stats.norm.pdf(observations[index], means[2], math.sqrt(variances[2]))

	### done initializing

	### start viterbi
	#probability matrix
	t1 = np.zeros((len(states_map), len(observations)))
	#accumulated state matrix
	t2 = np.zeros((len(states_map), len(observations)), dtype=np.int32)

	#probability of beginning states
	for state in range(len(states_map)):
		t1[state, 0] = large_pi[state] * emission_matrix[state, 0]
	
	for observation in range(1, len(observations)):
		for state in range(len(states_map)):
			probabilities = t1[:, observation-1] * transition_matrix[:, state] * emission_matrix[state, observation]
			#which state is most likely for previous observation?
			t1[state, observation] = np.max(probabilities)
			t2[state, observation] = np.argmax(probabilities)

	#best path 
	most_probable_path = np.zeros(len(observations), dtype=np.int32)
	#init backtracking, assign most probable last state to end of most_probable_path
	most_probable_path[-1] = np.argmax(t1[:, len(observations)-1])
	
	for observation in range(len(observations)-1, 0, -1):
		most_probable_path[observation-1] = t2[int(most_probable_path[observation]), observation]

	return most_probable_path

def run_forward_algorithm(observations):
	# observations: a numpy array of driving speeds
	# returns: floating point probability of observing these observations under the model
	
	# in other words: P(state at time t | all observations from 0 until t) – forward
	# does so by calculating belief steps from 0 to t (most likely state at 0 < k < t)

	large_pi = np.array([1/3, 1/3, 1/3])
	
	emission_matrix = np.zeros((3, len(observations)))
	for index in range(len(observations)):
		emission_matrix[0, index] = stats.norm.pdf(observations[index], means[0], math.sqrt(variances[0]))
		emission_matrix[1, index] = stats.norm.pdf(observations[index], means[1], math.sqrt(variances[1]))
		emission_matrix[2, index] = stats.norm.pdf(observations[index], means[2], math.sqrt(variances[2]))

	forward_trellis = np.zeros((len(states_map), len(observations)), dtype=np.float64)

	# add start state probability: 0 (x0) = 1/3
	forward_trellis[:,0] = np.multiply(large_pi, emission_matrix[:,0])
	
	### done initializing

	for observation in range(1, len(observations)):
		for state in range(len(states_map)):
			probabilities = forward_trellis[:, observation-1] * transition_matrix[:, state] * emission_matrix[state, observation]
			forward_trellis[state, observation] = np.sum(probabilities)

	forward_prob = np.sum(forward_trellis[:,-1])
	
	return forward_prob

def run_posterior_decoding(observations):
	# also called forward – backward algorithm (goes forward first, then smoothes going backward, from 0 < k < t, then multiplies the probabilities)

	# observations: a numpy array of driving speeds
	# returns: a 3 x m numpy array where m is the length of the observations array and
	# 		   arr[j][l] = P(pi_{l} = states[j] | observations)
	# note: No tests for this function are provided. Write your own to ensure correctness.

	large_pi = np.array([1/3, 1/3, 1/3])
	
	emission_matrix = np.zeros((3, len(observations)))
	for index in range(len(observations)):
		emission_matrix[0, index] = stats.norm.pdf(observations[index], means[0], math.sqrt(variances[0]))
		emission_matrix[1, index] = stats.norm.pdf(observations[index], means[1], math.sqrt(variances[1]))
		emission_matrix[2, index] = stats.norm.pdf(observations[index], means[2], math.sqrt(variances[2]))

	forward_matrix = np.zeros((len(states_map), len(observations)), dtype=np.float64)

	# add start state probability: 0 (x0) = 1/3
	forward_matrix[:,0] = np.multiply(large_pi, emission_matrix[:,0])
	
	### done initializing

	for observation in range(1, len(observations)):
		for state in range(len(states_map)):
			probabilities = forward_matrix[:, observation-1] * transition_matrix[:, state] * emission_matrix[state, observation]
			forward_matrix[state, observation] = np.sum(probabilities)

	#start backward algo
	backward_matrix = np.zeros((3, len(observations)))

	backward_matrix[:, len(observations)-1] = [1,1,1]
	#L = 99 
	#99 - 1
	for observation in range(len(observations)-2, -1, -1):
		for state in range(len(states_map)):
			#=> 98, 0 incl. 
			probabilities = transition_matrix[state, :] * emission_matrix[:, observation+1] * backward_matrix[:, observation+1]
			backward_matrix[state, observation] = np.sum(probabilities)

	#smoothing algo, multiply f * b, then normalize result
	result = np.zeros((3, len(observations)))
	
	result = np.multiply(forward_matrix, backward_matrix)
	result = result / np.sum(result, axis=0, keepdims=True)

	return result

def main():
	(observations, most_probable_states, observation_probs) = read_data()

	run_posterior_decoding(observations[0])

	for i in range(observations.shape[0]):
		if not np.allclose(most_probable_states[i], run_viterbi(observations[i])):
			raise Exception("run_viterbi() is incorrect for example {}".format(i + 1))
		
		if not math.isclose(observation_probs[i], run_forward_algorithm(observations[i]), rel_tol=1e-03):
			raise Exception("run_forward_algorithm() is incorrect for example {}".format(i + 1))
	
	print("No errors detected. Remember to write tests to check run_posterior_decoding.")

if __name__ == '__main__':
	main()