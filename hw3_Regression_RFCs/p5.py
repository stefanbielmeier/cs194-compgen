
"""
Problem 5: Training a variant effect predictor

Training a machine learning algorithm to predict the pathogenicity of missense.
Model will be trained on variants from ClinVar, a publicly accessible database of human variants and their associated phenotypes.
"""

from os import path
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pyBigWig


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
                
                #case 2: for variants with multiple RVIS gene matches: take the average of the RVIS scores (doesn't happen or I misunderstood)
                #GENES=BRCA2, BRCA1: take score, divide by 2
                #TODO
                
                #case 3 no RVIS score
                else: 
                    #
                    scores[i] = 50
            
            #case 3: for variants without RVIS score (no match): assign percentile of 50
            else:
                scores[i] = 50
    
    return np.hstack((variants, scores.reshape((-1,1))))

def plot_hist(feature_matrix):

    y = feature_matrix[:,-2:].astype(float)
    benign_rvis = np.reshape(y[(np.where(y[:,0] == 0)), 1], (-1,1))
    path_rvis = np.reshape(y[(np.where(y[:,0] == 1)), 1], (-1,1))

    plt.hist(benign_rvis, 100, label='benign')
    plt.hist(path_rvis, 100, label='pathogenic')
    plt.ylabel('score frequency')
    plt.xlabel('score')
    plt.legend()
    plt.title('hist of score')
    plt.show()
    

def get_oe(filepath):
    """
    takes file path for The o/e (observed/expected) ratio extraction from a text file

    returns dict of key value pairs – key is the gene, and value is the o/e score (oe_lof_upper_rank column – 35th column)
    """

    dict = {}

    with open(filepath) as file:
        for line in file:
            if line[0:4] != "gene":
                info = line.rstrip().split('\n')[0].split("\t")
                #position 0: gene
                #position 32: oe_lof_upper_rank
                
                try:
                    #value is number, not N/A
                    value = float(info[34])
                except ValueError:
                    value = info[34]
                dict[info[0]] = value
    
    return dict

def add_oe_scores(variants, oe):
    """
    Takes: variant matrix in VCF format, oe scores (dict – key (gene) - value(oe score) pairs)
    Returns: variant matrix in VCF format with an additional column for each variant that includes the applicable RVIS score
    """
    scores = np.zeros((variants.shape[0]))

    for i in range(variants.shape[0]):
        info = variants[i, 7].split(';')
        for index in range(len(info)):
            subinfo = info[index].split("=")
            #every variant probably has a GENE score
            if subinfo[0] == "GENES":
                #case 1: for variants with one GENE (direct match)
                if subinfo[1] in oe:
                    if oe[subinfo[1]] == 'NA':
                        scores[i] = 0
                    else: 
                        scores[i] = oe[subinfo[1]]
                
                #case 2: for variants with multiple RVIS gene matches: take the average of the RVIS scores (doesn't happen or I misunderstood)
                #GENES=BRCA2, BRCA1: take score, divide by 2
                #TODO

                else: 
                    #
                    scores[i] = 0
            
            #case 1: for variants without RVIS score (no match): assign half-value of 50
            else:
                scores[i] = 0
    
    return np.hstack((variants, scores.reshape((-1,1))))

def add_phast_cons(variants, filepath):
    """
    Takes in: filepath to bigWig file
    returns: feature (,N) 2D numpy array with probability of a nucleotide being part in a conserved region, given a variant matrix acc. to a VCF file
    """

    phast_cons = np.zeros(variants.shape[0])
    bw = pyBigWig.open(filepath)

    #chrom, #pos, #id
    for row_idx in range(variants.shape[0]):
        chrom = "chr" + str(variants[row_idx, 0])
        pos = int(variants[row_idx, 1])
        phast_cons[row_idx] = bw.values(chrom, pos, pos+1)[0]
    
    return phast_cons.reshape((-1,1))

def main():

    num_benign, num_pathg, variants = filter_clinVar('clinvar_missense.vcf')
    #print("Number of benign and pathogenic variants in ClinVar", num_benign, num_pathg)
    
    #train, val = random_split(variants, 0.8)

    #get last column of val / train (benign / pathogenic), get True / False Array, get values of val / train's last column that matching T/F array, get those dimensions!
    #print("val benign and path", np.shape(val[(np.where(val[:,-1] == '0')), -1])[1], np.shape(val[(np.where(val[:,-1] == '1')), -1])[1])
    #print("train benign and path", np.shape(train[(np.where(train[:,-1] == '0')), -1])[1], np.shape(train[(np.where(train[:,-1] == '1')), -1])[1])
    
    rvis = get_rvis_scores("RVIS_Unpublished_ExACv2_March2017.txt")
    feature1 = add_rvis_scores(variants, rvis)
    
    #plot_hist(feature1)

    oe = get_oe('gnomad.v2.1.1.lof_metrics.by_gene.txt')

    feature2 = add_oe_scores(variants, oe)
    #plot_hist(feature2)
    #print(variants[20000:20010, :])
    phast_cons = add_phast_cons(variants, "hg38.phastCons100way.bw")
    feature3 = np.hstack((variants, phast_cons))

    dataset = np.hstack((feature1[:,-1].reshape((-1,1)), feature2[:,-1].reshape((-1,1)), feature3[:,-1].reshape((-1,1)), feature1[:,-2].reshape((-1,1))))
    np.savetxt('dataset.csv', dataset)

if __name__ == '__main__':
	main()