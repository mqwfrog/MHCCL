
### MHCCL: Masked Hierarchical Cluster-wise Contrastive Learning for Multivariate Time Series - A PyTorch Version
Authors: Qianwen Meng, Hangwei Qian, Yong Liu, Yonghui Xu, Zhiqi Shen, Lizhen Cui
This work is accepted for publication in Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI 2023).

### Citation:
If you find any of the codes helpful, kindly cite our paper.

@misc{meng2022mhccl,
    title={MHCCL: Masked Hierarchical Cluster-wise Contrastive Learning for Multivariate Time Series},
    author={Qianwen Meng and Hangwei Qian and Yong Liu and Yonghui Xu and Zhiqi Shen and Lizhen Cui},
    year={2022},
    eprint={2212.01141},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

### Requirements for this project:
- Python ≥ 3.6
- PyTorch ≥ 1.4


### Packages that need to be installed:
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


### To perform unsupervised representation learning, use the following options:
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


### To perform downstream classification task based on the learned representations, use the following options:
<pre>
python classifier.py \
--dataset_name wisdm \
--pretrained experiment_wisdm/checkpoint_0149.pth.tar \
--lr 5 --batch_size 128 \
--dist_url 'tcp://localhost:10002' --multiprocessing_distributed --world_size 1 --rank 0 \
--id wisdm_linear_0149
</pre>


### Results:
- The experimental results will be saved in "experiment_{args.dataset_name}" directory by default 
- If you choose different partitions or masking strategies, the suffix will be added automatically such as
  "experiment_{args.dataset_name}_{args.mask_mode}_layer0_{args.dist_threshold}" 



### References:
Part of the augmentation transformation functions are adapted from

https://github.com/emadeldeen24/TS-TCC
https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
https://github.com/LijieFan/AdvCL/blob/main/fr_util.py

Part of the contrastive models are adapted from

https://github.com/salesforce/PCL
https://github.com/lucidrains/byol-pytorch
https://github.com/lightly-ai/lightly
https://github.com/emadeldeen24/TS-TCC

Loggers used in the repo are adapted from
https://github.com/emadeldeen24/TS-TCC



