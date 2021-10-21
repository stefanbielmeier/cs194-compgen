from functools import reduce
import numpy as np
import statsmodels.api as sm
from p2 import apply_benjamini_hochberg_correction, apply_bonferroni_correction
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PHENOTYPE_DATA = "phenotypes.npz"

def calc_dosages_matrix():
	dosages = []
	snps = 0
	with open("chr22_subsampled_snps.vcf") as file:
		#create the matrix
		for line in file:
			if (line[0] != "#"):
				snp = line.rstrip().split('\n')[0].split("\t")[9:]
				for idx in range(len(snp)):
					if snp[idx] == "0|0":
						snp[idx] = 0
					elif "0" in snp[idx]:
						snp[idx] = 1
					else:
						snp[idx] = 2
				
				snps += 1
				dosages.append(snp)
				
		dosages_matrix = np.array(dosages)
	return dosages_matrix.transpose()

def get_phenotype_data():
	'''
	Returns:
		probands: a numpy array (dtype: string) of proband ids
		phenotypes: a numpy array (dtype: int) of binary phenotypes
			phenotypes[i] describes the phenotype for probands[i]
	'''
	data = np.load(PHENOTYPE_DATA)
	probands = data["probands"]
	phenotypes = data["phenotypes"]
	return probands, phenotypes


def run_gwas(dosages, phenotypes):

	p_values = []
	for snp in range(len(dosages[0,:])):

		exog = sm.add_constant(dosages[:,snp])
		logit_model = sm.Logit(phenotypes, exog)
		logit_res = logit_model.fit(method="bfgs")
		
		if len(logit_res.pvalues) == 1:
			p_values.append(1.0)
		else:
			p_values.append(logit_res.pvalues[1])
		
	return p_values

def run_pca_gwas(dosages, phenotypes):
	
	#normalize dosage matrix along each column (SNP)
	normed_dosages = (dosages - np.mean(dosages, axis=1, keepdims=True)) / np.std(dosages, axis=1, keepdims=True)

	#run pca for each individual (sample)
	pca = PCA(n_components=3)
	reduced_dosages = pca.fit_transform(normed_dosages) #of dimension (2504, 3)

	"""
	logistic regression needs as input for each SNP
	[2504, 5] B0 (intercept), B1, PC1, PC2, PC3
	"""	
	p_values = []
	for snp in range(len(dosages[0,:])):
		exog = sm.add_constant(normed_dosages[:,snp])
		with_pca = np.append(exog, reduced_dosages, axis=1)
		logit_model = sm.Logit(phenotypes, with_pca)
		logit_res = logit_model.fit(method="bfgs")
		
		if len(logit_res.pvalues) == 1:
			p_values.append(1.0)
		else:
			p_values.append(logit_res.pvalues[1])
	
	return p_values
	
def plot_gwas(p_values):

	log_pvalues = -np.log10(p_values)

	chrom_position = np.arange(0, len(log_pvalues)) # snp file was sorted ascending by position in chromosome
	plt.plot(chrom_position, log_pvalues, ls='', marker='.')

	alpha = 0.05

	bonferroni_cutoff = -np.log10(alpha/len(p_values))
	rejects_bonf = apply_bonferroni_correction(p_values, alpha)
	plt.axhline(y = bonferroni_cutoff, color = 'r', linestyle = '-')
	
	rejects = apply_benjamini_hochberg_correction(p_values, alpha)

	benjamin_hochberg_cutoff = -np.log10(len(rejects[rejects == True]) * alpha / np.size(rejects))
	plt.axhline(y = benjamin_hochberg_cutoff, color = 'b', linestyle = '-')
	
	print("total no of snps", len(rejects))
	print("number of SNP H0 rejects (they are associated with phenotype) after boneferroni", len(rejects_bonf[rejects_bonf == True]))
	print("number of SNP H0 rejects (they are associated with phenotype) after Benjamin-Hochberg", len(rejects[rejects==True]))
	# it makes sense that bonferroni is stricter than benjamin hochberg
	plt.title("Manhattan Plot with Boneferroni (red) and Benjamin-Hochberg (blue) cutoff lines")
	plt.show()


def main():
	probands, phenotypes = get_phenotype_data()

	dosages = calc_dosages_matrix()

	p_values = run_gwas(dosages, phenotypes) # Problem 4a)
	plot_gwas(p_values)

	pca_p_values = run_pca_gwas(dosages, phenotypes) #Problem 4b)
	plot_gwas(pca_p_values)


if __name__ == '__main__':
	main()