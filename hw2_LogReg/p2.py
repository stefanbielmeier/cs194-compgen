import numpy as np

def apply_bonferroni_correction(pvalues, alpha):
	'''
	Args:
		pvalues: numpy array of p-values
		alpha: desired familywise error rate (FWER)

	Returns:
		rejects: a numpy array of booleans indicating whether the null hypothesis should be rejected
				 at the given alpha
	'''
	array_size = np.size(pvalues)
	helper = np.full(array_size, alpha/array_size)
	rejects = pvalues <= helper

	return rejects

def apply_benjamini_hochberg_correction(pvalues, alpha):
	'''
	Args:
		pvalues: numpy array of independent p-values
		alpha: desired false discovery rate (FDR)

	Returns:
		rejects: a numpy array of booleans indicating whether the null hypothesis should be rejected
				 at the given alpha
	'''
	running_threshold = np.arange(1,np.size(pvalues)+1) * alpha / np.size(pvalues)
	orig_pos = np.argsort(pvalues)

	unordered_rejects = np.sort(pvalues) <= running_threshold
	rejects = np.array([x for _, x in sorted(zip(orig_pos, unordered_rejects))])

	return rejects 

def main():
	#test case 1 bonferri
	test_pvalues = np.array([0.01, 0.02, 0.1, 0.2, 0.001, 0.005])
	test_alpha = 0.05
	expected_output = np.array([False, False, False, False, True, True])

	#test case 2 bonferri
	test_pvalues2 = np.array([0.02, 0.03, 0.2, 0.001, 0.1])
	test_alpha2 = 0.1
	expected_output2 = np.array([True, False, False, True, False])

	#Test Bonferri
	if (np.array_equal(apply_bonferroni_correction(test_pvalues, test_alpha), expected_output)):
		print("Bonferri Test 1 pass")
	else:
		print("Test 1 bonferri didn't pass!")
	if (np.array_equal(apply_bonferroni_correction(test_pvalues2, test_alpha2), expected_output2)):
		print("bonferri test 2 pass")
	else:
		print("Test 2 bonferri didn't pass!")

	#test case 1 benjamini hochberg
	test_q = 0.05
	hochberg_expected_output = [True, True, False, False, True, True]
	
	#test case 2 benjamini hochberg
	test_q2 = 0.1
	hochberg_expected_output2 = [True, True, False, True, False]
	
	if (np.array_equal(apply_benjamini_hochberg_correction(test_pvalues, test_q), hochberg_expected_output)):
		print("hochberg Test 1 pass")
	else:
		print("Test 1 hochberg didn't pass!")
	if (np.array_equal(apply_benjamini_hochberg_correction(test_pvalues2, test_q2), hochberg_expected_output2)):
		print("hochberg test 2 pass")
	else:
		print("Test 2 hochberg didn't pass!")

	#tests that fail in autograder for hochberg
	alpha = 0.05 
	t1 = [0.07, 0.02, 0.001, 0.09, 0.0004]
	t1_exp = [False, True, True, False, True]
	t2 = [0.038, 0.15, 0.001, 0.02, 0.25]
	t2_exp = [False, False, True, True, False]
	print("autograder 1:", t1_exp == apply_benjamini_hochberg_correction(t1, alpha))
	print("autograder 2:", t2_exp == apply_benjamini_hochberg_correction(t2, alpha))


if __name__ == '__main__':
	main()