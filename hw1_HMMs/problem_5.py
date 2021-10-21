import numpy as np
import time

states = {
	"real": 0,
	"fake": 1
}

def get_data():
	with open("problem_5_data.txt") as f:
		rolls = f.readline().strip()
		return [int(r) for r in rolls]

def print_parameters(transition_matrix, emission_1, emission_2):
	print("Transition probability matrix: {}".format(transition_matrix))
	print("Emission probabilities for dice 1: {}".format(emission_1))
	print("Emission probabilities for dice 2: {}".format(emission_2))

def run_backward(observations, transition_matrix, emission_matrix):
	backward_matrix = np.zeros((len(states), len(observations)))

	backward_matrix[:, -1] = [1,1]
	#L = 99 
	#99 - 1
	for observation in range(len(observations)-2, -1, -1):
		for state in range(len(states)):
			#=> 98, 0 incl. 
			probabilities = transition_matrix[state, :] * emission_matrix[:, (observations[observation+1]-1)] * backward_matrix[:, observation+1]
			backward_matrix[state, observation] = np.sum(probabilities)
		#normalize with scaling factor (http://www.cs.cmu.edu/~roni/11661/2017_fall_assignments/shen_tutorial.pdf), alternative to logsumexp trick
		backward_matrix[:, observation] = backward_matrix[:, observation] * (1 / np.sum(backward_matrix[:, observation]))
	
	return backward_matrix

def run_forward(observations, start_state_probs, emission_matrix, transition_matrix):

	#problem: this is underflowing
	forward_matrix = np.zeros((len(states), len(observations)), dtype=np.float64)

	# add start state probability: 0 (x0) = 1/2
	forward_matrix[:,0] = np.multiply(start_state_probs, emission_matrix[:,0])
	
	### done initializing

	for observation in range(1, len(observations)):
		for state in range(len(states)):
			probabilities = forward_matrix[:, observation-1] * transition_matrix[:, state] * emission_matrix[state, observations[observation]-1]
			forward_matrix[state, observation] = np.sum(probabilities)
		#normalize with scaling factor (section 6, http://www.cs.cmu.edu/~roni/11661/2017_fall_assignments/shen_tutorial.pdf), alternative to logsumexp trick
		forward_matrix[:, observation] = forward_matrix[:, observation] * (1 / np.sum(forward_matrix[:, observation]))

	return forward_matrix

def e_step(transition_matrix, emission_matrix, rolls, initial_state_distribution):

	#probability of being in state j at t
	gamma = np.zeros((2, len(rolls)))

	forward = run_forward(rolls, initial_state_distribution, emission_matrix, transition_matrix)
	backward = run_backward(rolls, transition_matrix, emission_matrix)
	fb = np.multiply(forward, backward)

	#normalize with scaling factor (section 6, http://www.cs.cmu.edu/~roni/11661/2017_fall_assignments/shen_tutorial.pdf), alternative to logsumexp trick
	gamma = fb / np.sum(fb, axis=0)

	#Probability of i at t, and j at t+1, real -> real(0), real->fake(1), fake->real(2), fake->fake(3)
	xi = np.zeros((2, 2, len(rolls)-1))

	for from_state in range(len(states)):
		for to_state in range(len(states)):
			for roll in range(len(rolls)-1):
				xi[from_state, to_state, roll] = forward[from_state, roll] * transition_matrix[from_state,to_state] * emission_matrix[to_state, rolls[roll]-1] * backward[to_state, roll+1]

	return gamma, xi

def m_step(gamma, xi, rolls):
	
	#sum up over columns
	sum_gamma = np.sum(gamma, axis=1, keepdims=True)
	emission_matrix = np.zeros((2,6))
	for observation in range(len(rolls)):
		emission_matrix[:, rolls[observation]-1] += 1 * gamma[:, observation]
	emission_matrix = emission_matrix / sum_gamma

	#from row to col

	transition_matrix = np.zeros((2,2))
	for from_state in range(2):
		for to_state in range(2):
			transition_matrix[from_state,to_state] = np.sum(xi[from_state, to_state, :]) / np.sum(gamma[from_state, :]) 
	#normalize by sum of each row as conditional probabilities have to add up to 1 (row 0: sum(from A to A, from A to B) = 1)
	transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)
	
	return transition_matrix, emission_matrix

def run_baum_welch(rolls):
	# return:
	# 	- 2 x 2 transition probability matrix
	#	- 6-dimensional array containing emission probabilities for dice 1
	#	- 6-dimensional array containing emission probabilities for dice 2

	#initialize with uniform probabilities
	# a[row][col] = P(col | row), from row to col, with row = 0: fair, row = 1: loaded, col = 0: fair, col = 1: loaded
	transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
	
	#fair, equal emission probabilities
	#loaded, 6 is more likely (assumption)
	emission_matrix = np.vstack((np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]), np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.4])))
	
	initial_state_distribution = [0.5, 0.5]

	#iterate until convergence
	iteration_count = 0
	start_time = time.time()

	while True:
		old_transition_matrix = transition_matrix
		old_emission_matrix = emission_matrix

		gamma, xi = e_step(transition_matrix, emission_matrix, rolls, initial_state_distribution)
		a, b = m_step(gamma, xi, rolls)
		transition_matrix = a
		emission_matrix = b
		initial_state_distribution = np.sum(gamma, axis=1) / np.sum(gamma)

		iteration_count += 1
		#break if the magnitude of the change matrix is < 0.001 
		if np.linalg.norm(old_transition_matrix-a) < 0.001 and np.linalg.norm(old_emission_matrix-b) < 0.001:
			break
	
	end_time = time.time()
	print("done, total time elapsed:", (end_time-start_time)/60)
	print("number of iterations until convergence:", iteration_count)

	#return transition_matrix, em_dice1, em_dice2
	return transition_matrix, np.round(emission_matrix[0], decimals=4), np.round(emission_matrix[1], decimals=4)

def main():
	rolls = get_data()
	print('Start')
	transition_matrix_A, emission_1_A, emission_2_A = run_baum_welch(rolls)
	
	# Part A
	print("Part A")
	print_parameters(transition_matrix_A, emission_1_A, emission_2_A)
	print("")

	# Part B
	print("Part B")
	transition_matrix_B, emission_1_B, emission_2_B = run_baum_welch(rolls[: 1000])
	print_parameters(transition_matrix_B, emission_1_B, emission_2_B)
	

if __name__ == '__main__':
	main()