
"""
Problem 5: Training a variant effect predictor

Training a machine learning algorithm to predict the pathogenicity of missense.
Model will be trained on variants from ClinVar, a publicly accessible database of human variants and their associated phenotypes.
"""

import numpy as np
from sklearn.svm import SVC


#4 https://hgdownload.cse.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw


def filter_clinVar(vcffile_path):

    """
    takes: ClinVAR vcf file path
    
    returns ClinVAR vcf file as numpy matrix, filtered only for benign and pathogenic clinical significances,
    with a new column at the right end of the matrix to denote if it's benign (0) or pathogenic 1).
    """

    variants = []
    num_benign = 0
    num_pathogenic = 0
    with open(vcffile_path) as file:
		#create the matrix
        for line in file:
            if (line[0] != "#"):
                variant = line.rstrip().split('\n')[0]
                table_form = variant.split("\t")
                info = table_form[7].split(';')
                for index in range(len(info)):
                    subinfo = info[index].split("=")
                    if subinfo[0] == "CLNSIG":
                        if subinfo[1] == "Pathogenic":
                            table_form.append(1)
                            num_pathogenic += 1
                            variants.append(table_form)
                        elif subinfo[1] == "Benign":
                            table_form.append(0)
                            num_benign += 1
                            variants.append(table_form)
         
    return num_benign, num_pathogenic, np.array(variants)


def random_split(matrix, train_size):

    """
    Takes matrix, and train_size as decimal value between [0,1]

    Randomly splits the number of rows in a matrix into a training matrix that is train_size of the rows, 
    and a validation set of (1- train_size) of the rows

    Returns training and validation matrices
    """
    
    """ 
    Tested with 
    arr = np.arange(10).reshape((5, 2))
    print(random_split(arr, 0.8))
    """

    num_rows = matrix.shape[0]
    permutation = np.random.permutation(num_rows)
    train_idxs = permutation[: int(train_size * num_rows)]
    test_idxs = permutation[int(train_size * num_rows): num_rows]
    
    return matrix[train_idxs, :], matrix[test_idxs, :]

def main():

    num_benign, num_pathg, variants = filter_clinVar('clinvar_missense.vcf')
    print(num_benign, num_pathg)

    train, val = random_split(variants, 0.8)

    print("val benign and path", np.shape(val[(np.where(val[:,-1] == '0')), -1])[1], np.shape(val[(np.where(val[:,-1] == '1')), -1])[1])
    print("train benign and path", np.shape(train[(np.where(train[:,-1] == '0')), -1])[1], np.shape(train[(np.where(train[:,-1] == '1')), -1])[1])
    

if __name__ == '__main__':
	main()