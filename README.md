
# MHCCL: Masked Hierarchical Cluster-wise Contrastive Learning for Multivariate Time Series - a PyTorch Version [[Paper]](https://arxiv.org/abs/2212.01141) [[Code]](https://github.com/mqwfrog/MHCCL)
Authors: Qianwen Meng, Hangwei Qian, Yong Liu, Yonghui Xu, Zhiqi Shen, Lizhen Cui  
This work is accepted for publication in Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI 2023).  


## Citation:
If you find any of the codes helpful, kindly cite our paper.   

@misc{meng2022mhccl,  
    title={MHCCL: Masked Hierarchical Cluster-wise Contrastive Learning for Multivariate Time Series},  
    author={Qianwen Meng and Hangwei Qian and Yong Liu and Yonghui Xu and Zhiqi Shen and Lizhen Cui},  
    year={2022},  
    eprint={2212.01141},  
    archivePrefix={arXiv},  
    primaryClass={cs.LG}  
}  

## MHCCL Overview:
![image](https://github.com/mqwfrog/MHCCL/blob/main/MHCCL_overview.png)

  
## Abstract
Learning semantic-rich representations from raw unlabeled time series data is critical for downstream tasks such as classification and forecasting. Contrastive learning has recently shown its promising representation learning capability in the absence of expert annotations. However, existing contrastive approaches generally treat each instance independently, which leads to false negative pairs that share the same semantics. To tackle this problem, we propose MHCCL, a Masked Hierarchical Cluster-wise Contrastive Learning model, which exploits semantic information obtained from the hierarchical structure consisting of multiple latent partitions for multivariate time series. Motivated by the observation that fine-grained clustering preserves higher purity while coarse-grained one reflects higher-level semantics, we propose a novel downward masking strategy to filter out fake negatives and supplement positives by incorporating the multi-granularity information from the clustering hierarchy. In addition, a novel upward masking strategy is designed in MHCCL to remove outliers of clusters at each partition to refine prototypes, which helps speed up the hierarchical clustering process and improves the clustering quality. We conduct experimental evaluations on seven widely-used multivariate time series datasets. The results demonstrate the superiority of MHCCL over the state-of-the-art approaches for unsupervised time series representation learning.  


## Requirements for this project:
- Python ≥ 3.6
- PyTorch ≥ 1.4


## Packages that need to be installed:
- numpy
- sklearn
- openpyxl 
- torchvision
- random
- copy
- pandas
- matplotlib
- time
- collections
- scipy
- pynndescent
- builtins
- math
- shutil
- os
- sys
- warnings
- tqdm
- argparse
- tensorboard_logger 


## To perform unsupervised representation learning, please refer to the options below:
<pre>
python main.py \
--dataset_name wisdm \
--lr 0.03 \
--batch_size 128 \
--mlp --cos \ 
--layers 3 \
--posi 2 \
--negi 100 \
--posp 3 \
--negp 4 \
--protoNCE_only \ #If --flag is not entered, the default value is False. The True value is triggered when --flag is entered
--mask_layer0 \ #If --flag is not entered, the default value is False. The True value is triggered when --flag is entered
--mask_others \ #If --flag is not entered, the default value is False. The True value is triggered when --flag is entered
--replace_centroids \ #If --flag is not entered, the default value is False. The True value is triggered when --flag is entered
--mask_mode 'mask_proportion' \ #choices['mask_farthest', 'mask_threshold'(if use, specify the dist_threshold), 'mask_proportion'(if use, specify the proportion)]
--proportion 0.5 \
--dist_threshold 0.7 \
--dist_url 'tcp://localhost:10001' --multiprocessing_distributed --world_size 1 --rank 0
</pre>


## To perform downstream classification task based on the learned representations, please refer to the options below:
<pre>
python classifier.py \
--dataset_name wisdm \
--pretrained experiment_wisdm/checkpoint_0149.pth.tar \
--lr 5 --batch_size 128 \
--dist_url 'tcp://localhost:10002' --multiprocessing_distributed --world_size 1 --rank 0 \
--id wisdm_linear_0149
</pre>


## Results:
- The experimental results will be saved in "experiment_{args.dataset_name}" directory by default 
- If you choose different partitions or masking strategies, the suffix will be added automatically such as
  "experiment_{args.dataset_name}_{args.mask_mode}_layer0_{args.dist_threshold}" 



## References:
Part of the codes are referenced from  

https://github.com/emadeldeen24/TS-TCC  
https://github.com/salesforce/PCL  
https://github.com/mqwfrog/FINCH-Clustering  




