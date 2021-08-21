# Cancer_classification_paracancerous_tissues

## **Citation**

Baoshan Ma, Bingjie Chai, Jishuang Qi, Heng Dong, Di Li, Shuxin Liu, Fengju Song.Diagnostic classification of cancers using DNA methylation of paracancerous tissues (Under review)

## The datasets of the program
The data used in this research are collected from The Cancer Genome Atlas(TCGA) project and that are publicly available at https://portal.gdc.cancer.gov.

## The describe of the program
The program is divided into two sections saved in this repository.
1.machine_learning_models:six machine learning models are utilized to classify cancer stage(early or late) of KIRC patients based on DNA methylation data of paracancerous tissues.
2.tumor_specific_multiclass_classifier:we build a CpG-based tumor specific classifier using XGBoost algorithm that can accurately classify cancer type.The files in the repository can be uesd to construct tumor specific classifier based on TCGA datasets and validate the classifier developed using TCGA dataset on an independent GEO dataset. 
