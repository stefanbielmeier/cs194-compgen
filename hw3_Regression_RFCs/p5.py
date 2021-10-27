
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

        filtered_variants = np.array(variants)

    return num_benign, num_pathogenic, filtered_variants


def main():

    num_benign, num_pathg, variants = filter_clinVar('clinvar_missense.vcf')
    print(num_benign, num_pathg)

if __name__ == '__main__':
	main()