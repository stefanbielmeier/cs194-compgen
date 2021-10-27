
"""
Problem 5: Training a variant effect predictor

Training a machine learning algorithm to predict the pathogenicity of missense.
Model will be trained on variants from ClinVar, a publicly accessible database of human variants and their associated phenotypes.
"""

from os import path
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


#4 https://hgdownload.cse.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw


def filter_clinVar(filepath):

    """
    takes: ClinVAR vcf file path
    
    returns ClinVAR vcf file as numpy matrix, filtered only for benign and pathogenic clinical significances,
    with a new column at the right end of the matrix to denote if it's benign (0) or pathogenic 1).
    """

    variants = []
    num_benign = 0
    num_pathogenic = 0
    with open(filepath) as file:
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


def get_rvis_scores(filepath):
    """
    takes file path for Residual Variation Intolerance Score extraction of a text file

    returns dict of key value pairs – key is the gene, and value is the %RVIS[pop_maf_0.05%(any)], or RVIS score
    """

    dict = {}

    with open(filepath) as file:
        for line in file:
            if line[0:7] != "CCDSr20":
                info = line.rstrip().split('\n')[0].split("\t")
                #position 0: gene
                #position 3: %RVIS[pop_maf_0.05%(any)]
                dict[info[0]] = info[3]
    
    return dict

def add_rvis_scores(variants, rvis):
    """
    Takes: variant matrix in VCF format, rvis scores (dict – key (gene) - value(rvis score) pairs)
    Returns: variant matrix in VCF format with an additional column for each variant that includes the applicable RVIS score
    """
    scores = np.zeros((variants.shape[0]))

    for i in range(variants.shape[0]):
        info = variants[i, 7].split(';')
        for index in range(len(info)):
            subinfo = info[index].split("=")
            if subinfo[0] == "GENES":
                #case 2: for variants with one RVIS score (direct match)
                if subinfo[1] in rvis:
                    scores[i] = rvis[subinfo[1]]
                #case 3: for variants with multiple RVIS gene matches: take the average of the RVIS scores (doesn't happen or I misunderstood)
                #maybe TODO: normalize by variant count, if multiple variants are associated with the same gene
            
            #case 1: for variants without RVIS score (no match): assign percentile of 50
            else:
                scores[i] = 50
    print(scores)
    return np.hstack((variants, scores.reshape((-1,1))))

def main():

    num_benign, num_pathg, variants = filter_clinVar('clinvar_missense.vcf')
    #print("Number of benign and pathogenic variants in ClinVar", num_benign, num_pathg)
    
    #train, val = random_split(variants, 0.8)

    #get last column of val / train (benign / pathogenic), get True / False Array, get values of val / train's last column that matching T/F array, get those dimensions!
    #print("val benign and path", np.shape(val[(np.where(val[:,-1] == '0')), -1])[1], np.shape(val[(np.where(val[:,-1] == '1')), -1])[1])
    #print("train benign and path", np.shape(train[(np.where(train[:,-1] == '0')), -1])[1], np.shape(train[(np.where(train[:,-1] == '1')), -1])[1])
    rvis = get_rvis_scores("RVIS_Unpublished_ExACv2_March2017.txt")
    feature1 = add_rvis_scores(variants, rvis)

    print(rvis)
    y = feature1[:,-2:].astype(float)
    print(y[0:100,:])
    benign_rvis = np.reshape(y[(np.where(y[:,0] == 0)), 1], (-1,1))
    path_rvis = np.reshape(y[(np.where(y[:,0] == 1)), 1], (-1,1))

    plt.hist(benign_rvis, 100, label='benign')
    plt.hist(path_rvis, 100, label='pathogenic')
    plt.ylabel('RVIS score frequency')
    plt.xlabel('RVIS score')
    plt.legend()
    plt.title('hist of rvis csore')
    plt.show()

    #print("train benign and path", np.shape(train[(np.where(train[:,-1] == '0')), -1])[1], np.shape(train[(np.where(train[:,-1] == '1')), -1])[1])

if __name__ == '__main__':
	main()