# Bayesian_Differentiable_Architecture_Search_for_Fault_Diagnosis
## Author: Zheng Zhou
## Paper Link: https://ieeexplore.ieee.org/document/9419078
This repository is dedicated to provide users of interests with the ability to solve fault diagnosis problems using ***Bayesian Differentiable Architecture Search (BDAS)*** in our paper "Bayesian Differentiable Architecture Search for Efficient Domain Matching Fault Diagnosis" (Zhou, 2021).

## How to run the code

Bayesian Differentiable Architecture Search (BDAS)基于DARTS进行开发，
有三个主函数，依次依赖：
1、train_the_supernet.py
对超网络本身的参数进行训练，分为两个阶段（warmup和path dropout），
本函数的目的是减缓超网络这种参数共享策略带来的偏差问题，减缓搜索结果中非参算子（如残差连接）的富集问题。

2、optimize_the_evaluator.py
训练架构参数矩阵，得到架构参数的分布
本函数的目的，是获得候选子网络的分布，从而利用下一个主函数，采样得到候选子网络的集合

3、inference_bayesian_sample.py
从分布中采样子网络，并得到精度、稀疏度等指标。

## Notes
（1）本代码的任务是故障诊断，因此数据集为实验室自有的一维振动信号，数据集无法公开。
（2）本代码未进行细致调参，性能尚未最优，效果可能不是很稳定。


## Citation
If you found this work useful, consider citing us:
```
@ARTICLE{
BDAS, 
author={Zheng Zhou and Tianfu Li and Zilong Zhang and Zhibin Zhao and Chuang Sun and Ruqiang Yan* and Xuefeng Chen}, 
journal={IEEE Transactions on Instrumentation and Measurement}, 
title={Bayesian Differentiable Architecture Search for Efficient Domain Matching Fault Diagnosis}, 
year={2021}, 
}
```
