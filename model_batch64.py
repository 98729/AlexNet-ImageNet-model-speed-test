"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.

See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
# import horovod.torch as hvd

# 新增多节点运行DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import time


# define model parameters
NUM_EPOCHS = 1
BATCH_SIZE = 64
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0]  # GPUs to use
# modify this to point to your data directory
# INPUT_ROOT_DIR = 'alexnet_data_in'
TRAIN_IMG_DIR = '/work1/aicao/teapot/data/ImageNet227/train/'
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            # nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # 去除normalization测试时间
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),   # dropout probability
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.init_bias()  # initialize bias and weights

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)  # Gaussian distribution
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        # nn.init.constant_(self.net[4].bias, 1)
        # nn.init.constant_(self.net[10].bias, 1)
        # nn.init.constant_(self.net[12].bias, 1)

        nn.init.constant_(self.net[3].bias, 1)
        nn.init.constant_(self.net[8].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)

def parse_opt():
    parser = argparse.ArgumentParser(description='PyTorch DDP ImageNet2012 slurm training')
    parser.add_argument("--master_addr", required=True, type=str, default='localhost', help="Address of master, will default to localhost.")
    parser.add_argument('--master_port', required=True, type=str, default='29500', help="Port that master is listening on, will default to 29500.")
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--dist_url', type=str, help='distributed backend init_method')
    parser.add_argument('--world_size', type=int, default=1, help='world size of multi gpu/machine training')
    parser.add_argument('--local_rank', type=int, default=0, help='process rank')
    opt = parser.parse_args()
    return opt
    

if __name__ == '__main__':

    # time_start = time.time()

    # print the seed value
    seed = torch.initial_seed()
    # print('Used seed : {}'.format(seed))

    # tbwriter = SummaryWriter(log_dir=LOG_DIR)
    # print('TensorboardX summary writer created')

    opt = parse_opt()
    torch.distributed.init_process_group(backend="nccl", init_method=opt.dist_url, world_size=opt.world_size, rank=opt.local_rank)

    # create model
    # local_rank = opt.local_rank
    # torch.cuda.set_device(local_rank)
    # device = torch.device(f'cuda:{opt.local_rank}')
    device = torch.device("cuda:0")
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    
    # 构造DDP模型
    # alexnet = alexnet.cuda() # 提前把模型加载到GPU
    alexnet = DDP(alexnet, device_ids=DEVICE_IDS, output_device=opt.local_rank).to(device)
    # print(alexnet)
    # print('AlexNet created')

    # alexnet.apply(weights_init)

    # create dataset and data loader
    # data_time_1 = time.time()
    dataset = datasets.ImageFolder(os.path.join(TRAIN_IMG_DIR), transforms.Compose([
        # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.CenterCrop(IMAGE_DIM),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    # print('Dataset created')
    # 新增DistributedSampler
    train_sampler = DistributedSampler(dataset)
    dataloader = data.DataLoader(
        dataset=dataset,
        # shuffle=True,
        pin_memory=True,
        # num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE,
        sampler=train_sampler)
    # print('Dataloader created')
    # data_time_2 = time.time()
    # data_time = data_time_2 - data_time_1
    # print(data_time)

    # create optimizer
    # the one that WORKS
    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    # optimizer = optim.Adam(params=alexnet.parameters(), lr=LR_INIT, weight_decay=LR_DECAY)
    ### BELOW is the setting proposed by the original paper - which doesn't train....
    # optimizer = optim.SGD(
    #     params=alexnet.parameters(),
    #     lr=LR_INIT,
    #     momentum=MOMENTUM,
    #     weight_decay=LR_DECAY)
    # print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) # 学习率调整
    # print('LR Scheduler created')

    # start training!!
    # print('Starting training...')
    total_steps = 0
    time_start = time.time()
    for epoch in range(NUM_EPOCHS):

        

        # lr_scheduler.step()
        for imgs, classes in dataloader:
            imgs, classes = imgs.to(device), classes.to(device)
            
            # print(classes)

            # calculate the loss
            output = alexnet(imgs)
            # print(output)
            loss = F.cross_entropy(output, classes)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # # log the information and add to tensorboard
            # if total_steps % 10 == 0:
            #     with torch.no_grad():
            #         _, preds = torch.max(output, 1)
            #         accuracy = torch.sum(preds == classes)

            #         # print('Epoch: {} \tStep: {} \tLoss: {:.12f} \tAcc: {}'
            #         #     .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
            #         # tbwriter.add_scalar('loss', loss.item(), total_steps)
            #         # tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)

            # # print out gradient values and parameter average values
            # if total_steps % 100 == 0:
            #     with torch.no_grad():
            #         # print and save the grad of the parameters
            #         # also print and save parameter values
            #         # print('*' * 10)
            #         for name, parameter in alexnet.named_parameters():
            #             if parameter.grad is not None:
            #                 avg_grad = torch.mean(parameter.grad)
            #                 # print('\t{} - grad_avg: {}'.format(name, avg_grad))
            #                 # tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
            #                 # tbwriter.add_histogram('grad/{}'.format(name),
            #                         # parameter.grad.cpu().numpy(), total_steps)
            #             if parameter.data is not None:
            #                 avg_weight = torch.mean(parameter.data)
            #                 # s('\t{} - param_avg: {}'.format(name, avg_weight))
            #                 # tbwriter.add_histogram('weight/{}'.format(name),
            #                 #         parameter.data.cpu().numpy(), total_steps)
            #                 # tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)

            total_steps += 1

    time_end = time.time()
    time_total = time_end - time_start
    print('total time: ', time_total)
    print('imgs per second: ', BATCH_SIZE * total_steps / time_total)

        # # save checkpoints
        # checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
        # state = {
        #     'epoch': epoch,
        #     'total_steps': total_steps,
        #     'optimizer': optimizer.state_dict(),
        #     'model': alexnet.state_dict(),
        #     'seed': seed,
        # }
        # torch.save(state, checkpoint_path)

        # # save model
        # if dist.get_rank() == 0:
        #     torch.save(alexnet.module.state_dict(), "%d.ckpt" % epoch)
    
    # time_end = time.time()
    # time_total = time_end - time_start
    # print('total time: ', time_total)
    # print('imgs per second: ', BATCH_SIZE * total_steps / time_total)