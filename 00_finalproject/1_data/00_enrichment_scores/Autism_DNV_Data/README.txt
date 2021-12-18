----------------
Files Description
----------------

ASD_Discov_DNVs.txt -- Annotated de novo coding variants in ASD discovery cohorts 
ASD_Discov_Trios.txt -- Trios included in ASD discovery cohorts
ASD_Discov_DupPairs.txt -- Known MZ twin pairs and likely overlapping samples in ASD discovery 

hg19_mutrate_7mer_SPARK30K_Scaled -- Per gene haploid mutation rate of different variant classes. 

---------------------
Sample Table Columns
---------------------

FamID,IID
	De-identified family ID (ASD only) and sample ID.
FamHist
	(ASD discovery cohort only) Family history of ASD: multiplex or simplex.
	Multiplex families are defined by different criteria for different cohorts:
	In ASC, it is defined by multiple affected trio offspring in the same family.
	In MSSNG, defined by multiple affected siblings (based on db v5). 
	In SPARK, defined by the presence of at least one affected first degree relative 
	pair in the family. And In SSC, all families are simplex by ascertainment.
Sex
	Sex assigned at birth.
Pheno
	Clinical diagnosed or self-reported ASD affection status.		
CognitImpair
	(ASD discovery cohort only) Presence of cognitive impairment. In ASC, this is based 
	on a binary classification of intellectual disability. In SSC, this is derived from 
	Vineland score, yes if <=70 no otherwise. In SPARK, this is based on self-reported 
	cognitive impairment, intellectual disability or global developmental delay at 
	enrollment or in basic medial screen. Data is not available at the time of analysis 
	for samples from MSSNG.

---------------------
Variant Table Columns
---------------------

FamID,IID
	De-identified family ID (ASD only) and sample ID.
Sex
	Sex assigned at birth.
Pheno
	ASD affection status.
Chrom,Position,Ref,Alt
	Genomic coordinate (in hg19) of the variant and reference/alternative alleles.
VarID
	Variant ID: "Chrom:Position:Ref:Alt".
Context
	(SNV only) Tri-nucleotide sequence context.
GeneID
	Ensembl gene ID from GENCODE V19 Basic Set (Ensembl release 75).
	When a variant is mapped to multiple overlapping genes, gene IDs are semi-colon separated,
	and all other gene level annotations will appear in the same order of gene IDs and 
	separated by semi-colon.
HGNC
	Gene symbol based on HGNC 2018-07-22.
ExACpLI,LOEUFbin,Arisk
	Gene level metrics: ExAC pLI, gnomAD LOEUF decile, and A-risk prediction score.
GeneEff
	The gene level effect, defined as the most severe consequence among all protein
	coding transcripts. Annotations to multiple overlapping genes are semi-column separated. 
TransCount
	Number of transcripts that are annotated. All protein coding transcripts including up 
	and down-stream 5000 bp regions that overlap with the variant are included in annotation. 
TransIDs
	Ensembl IDs of annotated transcripts (from GENCODE V19 Basic Set).
	Different transcripts for each gene are comma separated, all other transcript level 
	annotations will appear in the same order of transcript IDs and separated by comma.
	Transcripts from different genes appear in the same order as gene IDs and are separated 
	by semi-column. 
TransEffs,cDNAChg,CodonChg,AAChg
	Transcript level annotations: functional consequences, cDNA, codon and amino acid changes.
REVEL,MPC,PrimateAI
	Missense pathogenicity prediction scores: REVEL, MPC, and PrimateAI
CADD13	
	PHRED-scaled CADD score v1.3 and truncated that only show values >= 20.
ExAC_ALL,gnomADexome_ALL
	Population allele frequencies from all samples in ExAC and gnomAD exomes. All variants 
	regardless their filtering flags were used to query allele frequencies.
LoF,LoF_filter,LoF_flags
    Loftee (v1) annotations using default parameters. This annotation is transcript specific.
    LoF is the final classification (HC or LC) for putative LoF variants in each transcript. 
    For variants that are classified as LC, failed filters for each transcript will be listed
    in LoF_filter. Additional flags will be listed in LoF_flags. See loftee doc for details.
pExt_GTExBrain,pExt_HBDR
	pExt metrics. It is operationally defined as the sum of expression levels of transcripts 
	that have the same functional consequences as GeneEff divided by the transcription levels 
	of all transcripts used in the annotation. For LoF variants, only transcripts affected
	by HC LoF (by loftee) are included in pExt calculation. Two sets of expression data 
	are used: GTEx v6 brain subset and Human Developmental Biology Resource (HBDR).
HGNCv24,DS_AG,DS_AL,DS_DG,DS_DL,DP_AG,DP_AL,DP_DG,DP_DL
	SpliceAI annotations: gene predicted to be influence by the variant (symbols from 
	GENCODE V24) and predicted distances and probabilities of splice site gain or loss events. 
	See SpliceAI doc for details.


---------------------
Mutation rate table columns
---------------------

Mu_Nonsense: 	Haploid mutation rate of stop gain SNVs
Mu_SpliceSite: 	... of canonical splice sites SNVs
Mu_Missense	... of all missense variants
Mu_Silent	... of synonymous variants
Mu_Dmis_MPC1:	... of predicted deleterious missense SNVs with MPC >=1
Mu_Dmis_MPC2:	... of predicted deleterious missense SNVs with MPC >=2
Mu_Dmis_CADD25-MPC2: ... of predicted deleterious missense SNVs with CADD>=25 and MPC<2
Mu_Dmis_PrimateAI0.8:	... of predicted deleterious missense SNVs with PrimateAI>=0.8 
Mu_Bmis_PrimateAIlt0.6:	... of predicted deleterious missense SNVs with PrimateAI>=0.6 
Mu_Dmis_REVEL0.5	:	... of predicted deleterious missense SNVs with REVEL>=0.5
Mu_Dmis_REVEL0.75:	... of predicted deleterious missense SNVs with REVEL>=0.75	
Mu_Bmis_REVELlt0.5:	... of predicted benign missense SNVs with REVEL<0.5 	
Mu_Dmis_CADD20...30: 	... of predicted deleterious missense SNVs with CADD>=20/.../30	
Mu_Bmis_CADDlt20...30: 	... of predicted benign missense SNVs with CADD<20/.../30	


