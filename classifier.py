import argparse
import builtins
import os
import random
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorboard_logger as tb_logger
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from dataloaderq.dataloaderm import data_generator
from sklearn.manifold import TSNE
from torchvision.models import resnet

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='Downstream Classification')

parser.add_argument('--dataset_name', type=str, default='wisdm',
                    help='name of dataset (default: wisdm')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=5., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--id', type=str, default='')
parser.add_argument('--low_dim', default=128, type=int,
                    help='feature dimension (default: 128)')


def ResNet18(low_dim=128, dataset_name='wisdm'):
    if dataset_name == 'HAR':
        in_channels = 9
    elif dataset_name == 'SHAR':
        in_channels = 3
    elif dataset_name == 'wisdm':
        in_channels = 3
    elif dataset_name == 'epilepsy':
        in_channels = 1
    elif dataset_name == 'FingerMovements':
        in_channels = 28
    elif dataset_name == 'PenDigits':
        in_channels = 2
    elif dataset_name == 'EigenWorms':
        in_channels = 6

    net = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], low_dim)
    if dataset_name == 'wisdm':
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=8,
                          stride=1, padding=4, bias=False)
    elif dataset_name == 'HAR':
        net.conv1 = nn.Conv2d(9, 64, kernel_size=8, stride=1, padding=4, bias=False)
    elif dataset_name == 'epilepsy':
        net.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=1, padding=43, bias=False)
    elif dataset_name == 'SHAR':
        net.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=1, padding=4, bias=False)
    else:
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.maxpool = nn.Identity()
    return net

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    args.tb_folder = 'Linear_eval/{}_tensorboard'.format(args.id)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)
        
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print('creating model')
    model = ResNet18(low_dim=args.low_dim, dataset_name=args.dataset_name)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    
    if args.gpu==0:
        logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    else:
        logger = None
        
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #     model.features = torch.nn.DataParallel(model.features)
        #     model.cuda()
        # else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.dataset_name == 'HAR':
        from config_files.HAR_Configs import Config as Configs
        configs = Configs()
        train_loader, val_loader = data_generator('data/HAR', configs, 'self_supervised')
    elif args.dataset_name == 'wisdm':
        from config_files.wisdm_Configs import Config as Configs
        configs = Configs()
        train_loader, val_loader = data_generator('data/wisdm', configs, 'self_supervised')
    elif args.dataset_name == 'epilepsy':
        from config_files.epilepsy_Configs import Config as Configs
        configs = Configs()
        train_loader, val_loader = data_generator('data/epilepsy', configs, 'self_supervised')
    elif args.dataset_name == 'SHAR':
        from config_files.SHAR_Configs import Config as Configs
        configs = Configs()
        train_loader, val_loader = data_generator('data/SHAR', configs, 'self_supervised')
    elif args.dataset_name == 'PenDigits':
        from config_files.PenDigits_Configs import Config as Configs
        configs = Configs()
        train_loader, val_loader = data_generator('data/PenDigits', configs, 'self_supervised')
    elif args.dataset_name == 'EigenWorms':
        from config_files.EigenWorms_Configs import Config as Configs
        configs = Configs()
        train_loader, val_loader = data_generator('data/EigenWorms', configs, 'self_supervised')
    elif args.dataset_name == 'FingerMovements':
        from config_files.FingerMovements_Configs import Config as Configs
        configs = Configs()
        train_loader, val_loader = data_generator('data/FingerMovements', configs, 'self_supervised')

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
            # train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        validate(val_loader, model, criterion, args, logger, epoch)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }) 
            if epoch == args.start_epoch:
                sanity_check(model.state_dict(), args.pretrained)


def plot_confusion_matrix(cm, labels_name, title):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for first_index in range(len(cm)):
        for second_index in range(len(cm[first_index])):
            plt.text(first_index, second_index, cm[first_index][second_index])


def plot_embedding(data, label, title, args):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)

    if args.dataset_name=='SHAR':
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], int(label[i]), color=plt.cm.tab20(label[i] / 20), fontdict={'weight': 'bold', 'size': 7})
        plt.xticks()
        plt.yticks()
        plt.title(title, fontsize=14)
    else:
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], int(label[i]), color=plt.cm.tab10(label[i] / 10),
                     fontdict={'weight': 'bold', 'size': 7})
        plt.xticks()
        plt.yticks()
        plt.title(title, fontsize=14)
    return fig

args = parser.parse_args()
representations_for_eb=np.empty(shape=[0,args.low_dim])

# tsne embeddings
outs_for_eb = np.array([])  # predict labels
trgs_for_eb = np.array([])  # true labels


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()

    for i, (data, target, aug1, aug2, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.gpu is not None:
            data = data.unsqueeze(3)
            aug1 = aug1.unsqueeze(3)
            aug2 = aug2.unsqueeze(3)
            data, target = data.float().cuda(args.gpu, non_blocking=True), target.long().cuda(args.gpu, non_blocking=True)
            series, aug2 = aug1.float().cuda(args.gpu, non_blocking=True), aug2.float().cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(series)
        loss = criterion(output, target)

        # record loss
        losses.update(loss.item(), series.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, logger, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        outs = np.array([])  # predict labels
        trgs = np.array([])  # true labels
        representations = np.empty(shape=[0, args.low_dim])

        for i, (data, target, aug1, aug2, index) in enumerate(val_loader):
            data = data.unsqueeze(3)
            aug1 = aug1.unsqueeze(3)
            aug2 = aug2.unsqueeze(3)
            data, target = data.float().cuda(non_blocking=True), target.long().cuda(non_blocking=True)
            series, aug2 = aug1.float().cuda(non_blocking=True), aug2.float().cuda(non_blocking=True)

            # compute output
            output = model(series)
            loss = criterion(output, target)

            # record loss
            losses.update(loss.item(), series.size(0))
          
            representations = np.concatenate((representations, output.cpu().detach().numpy()), axis=0)
            pred_label = torch.argmax(output.cpu(), dim=1)
            outs = np.append(outs, pred_label.cpu().detach().numpy())
            trgs = np.append(trgs, target.cpu().detach().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        outs_for_eb = outs
        trgs_for_eb = trgs
        representations_for_eb = representations

        """compute confusion matrix + save classification_report"""
        print('Starting to compute confusion matrix...')
        cm = confusion_matrix(outs_for_eb, trgs_for_eb)

        print('Starting to save classification performance...')
        r = classification_report(trgs_for_eb, outs_for_eb, digits=6, output_dict=True)
        df = pd.DataFrame(r)
        df["cohen"] = cohen_kappa_score(trgs_for_eb, outs_for_eb)
        df.to_csv(f'performance_{args.dataset_name}.csv', mode='a')

        if args.dataset_name == 'HAR':
            labels_name = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']  # HAR
        elif args.dataset_name == 'epilepsy':
            labels_name = ['epilepsy', 'not epilepsy']  # epilepsy
        elif args.dataset_name == 'SHAR':
            labels_name = ['StandingUpFS', 'StandingUpFL', 'Walking', 'Running', 'GoingUpS', 'Jumping', 'GoingDownS',
                           'LyingDownFS', 'SittingDown', 'FallingForw', 'FallingRight', 'FallingBack',
                           'HittingObstacle', 'FallingWithPS', 'FallingBackSC', 'Syncope', 'FallingLeft']  # SHAR
        elif args.dataset_name == 'wisdm':
            labels_name = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']  # wisdm
        elif args.dataset_name == 'PenDigits':
            labels_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        elif args.dataset_name == 'EigenWorms':
            labels_name = ['wild-type', 'goa-1', 'unc-1', 'unc-38', 'unc-63']
        elif args.dataset_name == 'FingerMovements':
            labels_name = ['Left', 'Right']
      
        if epoch == 99:
            ### plot confusion matrix
            print('Starting to plot confusion matrix...')
            plot_confusion_matrix(cm, labels_name, f"{args.dataset_name}--- Confusion Matrix")
            plt.subplots_adjust(bottom=0.2, left=0.2)
            plt.savefig(f'cm_{args.dataset_name}.png', format='png', bbox_inches='tight')

            ### TSNE Embeddings of representations
            print('Starting to compute t-SNE Embeddings...')
            ts = TSNE(perplexity=40, n_components=2, init='pca', random_state=0 , n_iter=3000)
            result = ts.fit_transform(representations_for_eb)
            fig = plot_embedding(result, trgs_for_eb, f't-SNE Embeddings of Time Series Representations---Dataset: {args.dataset_name}', args) 
            fig.tight_layout()
            plt.savefig(f'eb_{args.dataset_name}.png', format='png', bbox_inches='tight')



def save_checkpoint(state, filename='checkpoint_clf.pth.tar'):
    torch.save(state, filename)


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue
        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
