import itertools
import numpy as np
import time
import matplotlib.pyplot as plt


TRAIN_DATA = "problem_2_train_data.txt"
VAL_DATA = "problem_2_val_data.txt"

def load_data(input_path):
    # Loads promoter and negative sequences from the input_path.
    # DO NOT modify this function.

    promoter_sequences, negative_sequences = [], []
    with open(input_path) as f:
        for line in f:
            seq, clazz = line.strip().split()
            if clazz == "1":
                promoter_sequences.append(seq)
            elif clazz == "0":
                negative_sequences.append(seq)
            else:
                raise Exception("All class values should be either 0 or 1.")

    return promoter_sequences, negative_sequences

def train_markov_model(sequences, k):
    # Fits a Markov model where each state is a substring of size k.
	# These states are overlapping. So, if a sequence started with "ACTGA"
	# with k = 3, the first few states would be ["ACT", "CTG", "TGA", ...].
	#
	# returns: 
	#	- states: an ordered list of all possible states in the Markov model
	#	- transition_matrix: a probability matrix (2D numpy array) with size 4^k by 4^k such that
	# 	                     transition_matrix[row][col] = P(pi_{i + 1} = state[col] | pi_{i} = state[row])
	#							* in the above notation, pi_{i} denotes the ith state in the sequence

    states = list(map("".join, itertools.product("ACTG", repeat=k)))
    state_indexes = {state: index for index, state in enumerate(states, start=0)}
    
    # do not add pseudo-count r = 1 to the transition matrix because auto-grader will fail...
    transition_matrix = np.zeros((4**k, 4**k))
    
    #start at index = 0 and look at the transition which is length k*2
    #last transition to look at: 2*k
    for sequence in sequences:
        for index in range(0, len(sequence)-k):
            from_state = sequence[index:index+k]
            to_state = sequence[index+1:index+1+k]
            from_index = state_indexes[from_state]
            to_index = state_indexes[to_state]
            transition_matrix[from_index,to_index] += 1

    row_count = np.sum(transition_matrix, axis=1, keepdims=True)
    transition_matrix = transition_matrix / row_count
    
    return states, transition_matrix
    
def get_log_odds_ratio(seq, states, k, promoter_transition_matrix, negative_transition_matrix):
    # returns: log { P(sequence | promoter sequence model) / P(sequence | negative sequence model) }
    # assume that all first states are equally likely. That is, P(pi_{0} = state) = 1 / 4^k for all states

    states_dict = {state: index for index, state in enumerate(states, start=0)}

    log_odds = np.log(1/4**k)

    for index in range(0, len(seq)-k):
        from_state = seq[index:index+k]
        to_state = seq[index+1:index+1+k]
        from_index = states_dict[from_state]
        to_index = states_dict[to_state]
        log_odds += np.log(promoter_transition_matrix[from_index,to_index]/negative_transition_matrix[from_index,to_index])

    return log_odds

def get_accuracy(promoter_sequences, negative_sequences, states, k, 
                 promoter_transition_matrix, negative_transition_matrix):
    # Determine our model's accuracy on the given sequences.
    # Per our model, we classify a sequence as coming from a promoter iff it has a log odds ratio > 0.
    
    accuracy = np.longfloat(0)
    total_number_of_seq = len(promoter_sequences) + len(negative_sequences)
    correctly_identified_promoters = 0
    correctly_identified_negative = 0
    
    for seq in promoter_sequences:
        if get_log_odds_ratio(seq, states, k, promoter_transition_matrix, negative_transition_matrix) > 0:
            correctly_identified_promoters += 1

    for seq in negative_sequences:
        if get_log_odds_ratio(seq, states, k, promoter_transition_matrix, negative_transition_matrix) <= 0:
            correctly_identified_negative += 1
    
    accuracy = accuracy + ((correctly_identified_negative + correctly_identified_promoters) / total_number_of_seq)
    
    return accuracy

def main():
    train_promoter_sequences, train_negative_sequences = load_data(TRAIN_DATA)
    val_promoter_sequences, val_negative_sequences = load_data(VAL_DATA)
    start_time = time.time()
    
    val_accuracies = []

    for k in range(1,6):
        states, promoter_transition_matrix = train_markov_model(train_promoter_sequences, k)
        _, negative_transition_matrix = train_markov_model(train_negative_sequences, k)
        print("k is", k, "done training!")
        train_accuracy = get_accuracy(train_promoter_sequences, train_negative_sequences, states, k, 
                promoter_transition_matrix, negative_transition_matrix)
        val_accuracy = get_accuracy(val_promoter_sequences, val_negative_sequences, states, k, 
                promoter_transition_matrix, negative_transition_matrix)
        val_accuracies.append(val_accuracy)
        print("k = {}, train accuracy = {}, val accuracy = {}".format(k, train_accuracy, val_accuracy))

    end_time = time.time()
    print("done, total time elapsed:", (end_time-start_time)/60)

    plt.plot(np.array(range(1,6)), val_accuracies)
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.show()


if __name__ == '__main__':
    main()