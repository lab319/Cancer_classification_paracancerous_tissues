# User Guide

## Contact:
Baoshan Ma(mabaoshan@dlmu.edu.cn)

Bingjie Chai(qfcbj0612@163.com)

## Citation

Baoshan Ma, Bingjie Chai, Jishuang Qi, Heng Dong, Pengcheng Wang, Tong Xiong, Yi Gong, Di Li, Shuxin Liu, Fengju Song.Diagnostic classification of cancers using DNA methylation of paracancerous tissues (Under review)

## 1.Introduction
We provided a python program to build a classification model for separating early stage and late stage cancers. The classification model can then be applied to a new dataset. If you find the program useful, please cite the above reference. 

## 2.Software requirement
python 3.7 cersion

## 3.The datasets of the program
DNA methylation data used in this research were collected from The Cancer Genome Atlas(TCGA) project and that are publicly available at https://portal.gdc.cancer.gov. The 'input data' file contains 'data.txt' and 'label.txt'. Rows and columns of the 'data.txt' correspond to samples and CpG sites, respectively.

## 4.How to use our program and obtain output metrics
1.In the 'input data' folder, you need to upload a txt file of DNA methylation data of paracancerous tissues and name it 'data.txt'. Besides, you need upload a txt file of sample label(early stage or late stage) and name it 'label.txt'. For 'data.txt', we recommend that you can set each row as a sample and each column as a CpG site.

2.Now, you can run the model in the 
