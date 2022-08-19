
### MHCCL: Masked Hierarchical Cluster-wise Contrastive Learning for Multivariate Time Series - A PyTorch Version


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
--pcl_r 4 \
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




