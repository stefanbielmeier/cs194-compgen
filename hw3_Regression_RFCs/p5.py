
"""
Problem 5: Training a variant effect predictor

Training a machine learning algorithm to predict the pathogenicity of missense.
Model will be trained on variants from ClinVar, a publicly accessible database of human variants and their associated phenotypes.
"""
import math
from os import path
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pyBigWig

#Random forest model built with brainome
from rf_model import predict


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
                info = table_form[7].split(";")
                for index in range(len(info)):
                    subinfo = info[index].split("=")
                    if subinfo[0] == "CLNSIG":
                        if subinfo[1] == "Pathogenic":
                            table_form.append(1)
                            num_pathogenic += 1
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

def match_rvis_scores(variants, rvis_per_gene, infocol_idx):
    """
    Takes: variant matrix in VCF format, rvis scores (dict – key (gene) - value(rvis score) pairs), and an integer for the column number in the
    matrix that carries the info

    Returns: 2D numpy array of format (,N) with the applicable RVIS score in the same order as the variants input 
    """
    scores = np.zeros((variants.shape[0]))

    for i in range(variants.shape[0]):
        info = variants[i, infocol_idx].split(';')
        for index in range(len(info)):
            subinfo = info[index].split("=")
            if subinfo[0] == "GENES":
                #case 1: for variants with one gene (direct match with RVIS dict)
                if subinfo[1] in rvis_per_gene:
                    scores[i] = rvis_per_gene[subinfo[1]]
                
                #case 2: for variants with multiple genes: take the average of the RVIS scores
                #GENES=BRCA2, BRCA1: take score, divide by 2
                #if score is 0 (genes not found in dict, assign 50)
                elif "," in subinfo[1]:
                    genes = subinfo[1].split(",")
                    sum = 0
                    for gene in genes:
                        if gene in rvis_per_gene:
                            sum += float(rvis_per_gene[gene])
                    #no match for multiple variants
                    if sum == 0:
                        scores[i] = 50
                    else:
                        scores[i] = sum / len(genes)

                #case 3 no RVIS score for single gene
                else: 
                    scores[i] = 50
            
            #case 4: variant doesn't have gene info
            else:
                scores[i] = 50
    
    return scores.reshape((-1,1))

def plot_hist(matrix, y_col, feature):
    """
    Takes
        matrix: the matrix
        feature: 2D array of shape (N,1) of a feature.
        y_col: index of column that indicates the results

    Plots hist
    """
    feature = feature.astype(float)[:,0]
    y_values = matrix[:,y_col].astype(int)

    benign = feature[(np.where(y_values == 0))]
    pathogenic = feature[(np.where(y_values == 1))]

    plt.hist(benign, 100, label='benign')
    plt.hist(pathogenic, 100, label='pathogenic')
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

    oe_dict = {}

    with open(filepath) as file:
        for line in file:
            if line[0:4] != "gene":
                info = line.rstrip().split('\n')[0].split("\t")
                #position 0: gene
                #position 34: oe_lof_upper_rank
                
                try:
                    #value is number, not N/A
                    value = float(info[34])
                #if value is NA, don't add anything
                except ValueError:
                    continue
                oe_dict[info[0]] = value

    return oe_dict

def match_oe_scores(variants, oe, infocol_idx):
    """
    Takes: variant matrix in VCF format, oe scores (dict – key (gene) - value(oe score) pairs)
    Returns: variant matrix in VCF format with an additional column for each variant that includes the applicable RVIS score
    """
    scores = np.zeros((variants.shape[0]))
    middle_rank = (min(oe.values()) + max(oe.values())) / 2

    for i in range(variants.shape[0]):
        info = variants[i, infocol_idx].split(';')
        for index in range(len(info)):
            subinfo = info[index].split("=")
            #every variant probably has a GENE score
            if subinfo[0] == "GENES":
                #case 1: for variants with one GENE that has a direct match with the RVIS_dict
                if subinfo[1] in oe:
                    scores[i] = oe[subinfo[1]]
                
                #case 2: for variants with multiple GENES matches: take the average of the OE
                #GENES=BRCA2, BRCA1: take score, divide by 2
                elif "," in subinfo[1]:
                    genes = subinfo[1].split(",")
                    sum = 0
                    for gene in genes:
                        if gene in oe:
                            sum += oe[gene]
                    #no match for multiple variants
                    if sum == 0:
                        scores[i] = middle_rank
                    else:
                        scores[i] = sum / len(genes)

                #case 3 no OE score for single gene
                else: 
                    scores[i] = middle_rank
            
            #case 4: variant doesn't have gene info
            else:
                scores[i] = middle_rank
    
    return scores.reshape((-1,1))

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
        if math.isnan(bw.values(chrom, pos, pos+1)[0]):
            print(chrom, pos)
            #22	18939695	.	G	A	.	.	GENES=AC007326.4
            # we don't know the probability..., not in bigwig
            phast_cons[row_idx] = 0.5
        else:
            phast_cons[row_idx] = bw.values(chrom, pos, pos+1)[0]
    
    return phast_cons.reshape((-1,1))

def plot_roc(val):

    features = val[:,0:-1].astype(float)
    y_true = val[:,-1].astype(int)

    probs = predict(features, remap=False, return_probabilities=True)
    pos_probs = probs[:,1]

    plt.figure()
    lw = 2

    fpr, tpr, _ = roc_curve(y_true, pos_probs)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw = lw,
			label='ROC curve for random forest' + '(area = %.3f)' % auc_score)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Random Forest Class at 20 percent random val data')
    plt.legend(loc="lower right")
    plt.show()


def problem5_train():

    num_benign, num_pathg, variants = filter_clinVar('clinvar_missense.vcf')
    #print("Number of benign and pathogenic variants in ClinVar", num_benign, num_pathg)

    variants_train, variants_val = random_split(variants, 0.8)
    #get last column of val / train (benign / pathogenic), get True / False Array, get values of val / train's last column that matching T/F array, get those dimensions!
    print("val benign and path", np.shape(variants_val[(np.where(variants_val[:,-1] == '0')), -1])[1], np.shape(variants_val[(np.where(variants_val[:,-1] == '1')), -1])[1])
    print("train benign and path", np.shape(variants_train[(np.where(variants_train[:,-1] == '0')), -1])[1], np.shape(variants_train[(np.where(variants_train[:,-1] == '1')), -1])[1])
    
    rvis = get_rvis_scores("RVIS_Unpublished_ExACv2_March2017.txt")
    feature1 = match_rvis_scores(variants, rvis, infocol_idx = 7)
    
    #plot_hist(variants, feature = feature1, y_col=-1)

    oe_dict = get_oe('gnomad.v2.1.1.lof_metrics.by_gene.txt')

    feature2 = match_oe_scores(variants, oe_dict, infocol_idx = 7)
    #plot_hist(variants, feature = feature2, y_col=-1)

    feature3 = add_phast_cons(variants, "hg38.phastCons100way.bw")

    dataset = np.hstack((feature1, feature2, feature3, variants[:,-1].reshape((-1,1))))

    np.savetxt('dataset.csv', dataset.astype(str), fmt="%s", delimiter=",")

    _, val = random_split(dataset, 0.8)

    plot_roc(val)

def main():

    #for problems 5a – f), uncomment:
    #problem5_train()

    #for problem 5g)
    _ , _, variants = filter_clinVar('test_set.vcf')

    rvis = get_rvis_scores("RVIS_Unpublished_ExACv2_March2017.txt")
    feature1 = match_rvis_scores(variants, rvis, infocol_idx = 7)
    
    #plot_hist(variants, feature = feature1, y_col=-1)

    oe_dict = get_oe('gnomad.v2.1.1.lof_metrics.by_gene.txt')

    feature2 = match_oe_scores(variants, oe_dict, infocol_idx = 7)
    #plot_hist(variants, feature = feature2, y_col=-1)

    feature3 = add_phast_cons(variants, "hg38.phastCons100way.bw")

    dataset = np.hstack((feature1, feature2, feature3)).astype(float)
    print(dataset)
    for idx in range(dataset.shape[0]):
        for jdex in range(dataset.shape[1]):
            if math.isnan(dataset[idx,jdex]) == True:
                print(dataset[idx, jdex])
                print(idx, jdex)

    probs = predict(dataset, remap=False, return_probabilities=True)[:,1]
    print(probs)

    with open('test_set.predictions', 'w') as file:
        for prob in probs:
            file.write(str(prob))
            file.write('\n')

if __name__ == '__main__':
	main()