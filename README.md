# Bayesian_Differentiable_Architecture_Search_for_Fault_Diagnosis
## Author: Zheng Zhou, Ruqiang Yan*
## Paper Link: https://ieeexplore.ieee.org/document/9419078
This repository is dedicated to provide users of interests with the ability to solve fault diagnosis problems using ***Bayesian Differentiable Architecture Search (BDAS)*** in our paper "Bayesian Differentiable Architecture Search for Efficient Domain Matching Fault Diagnosis" (Zhou, 2021).

## How to run the code
BDAS is development version of the famous Differentiable Architecture Search (DARTS) method.  
The codes of BDAS has three sequential main functions:  
1. train_the_supernet.py  
This code is used to train the weights of hyper-network, using warmup and path-dropout strategies to alleviate the co-adaptation problem of hyper-network and obtain an intuitively fair hyper-network.  
2. optimize_the evaluator.py  
This code is used to optimize the architecture weights of hyper-network, using variational inference to estimate the distribution of architecture. In this stage, a scale mixture prior is used to encourage sparsity of architecture.  
3. inference_bayesian_sample.py  
This code is used to obtain a candidate architectures set by sampling from the previous distribution, and calculate their corresponding performance, i.e., accuracy and sparsity.

## Notes
(1) The task in this work is for fault diagnosis. As the dataset is private, you should modify BDAS in a public dataset.  
(2) The hyperparameters of BDAS has not been carefully adjuested, so its performance may not be optimal and stable.  

## Citation
If you found this work useful, consider citing us:
```
@ARTICLE{
BDAS2021, 
title={Bayesian Differentiable Architecture Search for Efficient Domain Matching Fault Diagnosis},
author={Zheng Zhou and Tianfu Li and Zilong Zhang and Zhibin Zhao and Chuang Sun and Ruqiang Yan* and Xuefeng Chen}, 
journal={IEEE Transactions on Instrumentation and Measurement}, 
year={2021}, 
}
```
