from __future__ import print_function
import os
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

import mlflow
import mlflow.pytorch

from utils import progress_bar
from utils import get_model_complexity_info
from folder2lmdb import ImageFolderLMDB
from torch.utils.tensorboard import SummaryWriter


#-------------------------------------------------------------Arguments--------------------------------------------
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch IMAGENET Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-5 , type=float, help='Weight Decay')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resumebestloss', action='store_true', help='resume from best loss checkpoint')
parser.add_argument('--optim', type=str, help='Optimizer', required=True)
parser.add_argument('--name', type=str, help='TensorboardName and Name to save the models', required=True)
parser.add_argument('--epoch', type=int, help='Number of Epochs to Train', required=True)
parser.add_argument('--decay', action='store_true', help='Decay Learning Rate')
args = parser.parse_args()

best_acc = 0 
start_epoch = 0
best_loss = 0 
writer = SummaryWriter(comment=args.name)
#------------------------------------------------------------Data Loading-------------------------------------------
print('==> Preparing data..')
traindir = '/media/iitm/data1/Surya/train.lmdb'
valdir   = '/media/iitm/data1/Surya/val.lmdb'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


train_transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])
train_dataset = ImageFolderLMDB(traindir, train_transform)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,num_workers=4, pin_memory=True)

val_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize])
val_dataset   = ImageFolderLMDB(valdir, val_transform)
val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)


#-------------------------------------------------------------Model Loading and Modification-----------------------------------------------
#-------------------------------------------------------------Edit this Part Before Training ----------------------------------------------
print('==> Building model..')
from models.mobilenetv3 import MobileNetV3
from models.mobilenetv3 import BottleNeck_Sys_Friendly
from models.mobilenetv3 import Flatten

net = MobileNetV3(mode='small', dropout=0)
state_dict = torch.load('./pretrainedmodels/mobilenetv3_small_67.4.pth.tar')
net.load_state_dict(state_dict, strict=True)

# l = []
# for name, child in net.named_children():
#     if name == 'features':
#         for x,y in child.named_children():
#             if x == '11':
#                 l.append(BottleNeck_Sys_Friendly(96, 96, 5, 1, 576, True, 'HS'))
#             else:
#                 l.append(y)
#     elif name == 'classifier':
#         l.append(Flatten())
#         l.append(child)

# new_model = nn.Sequential(*l)
# net = new_model
# l = []
# for name, child in net.named_children():
# 	if name == '10':
# 		l.append(BottleNeck_Sys_Friendly(96, 96, 5, 1, 576, True, 'HS'))
# 	else:
# 		l.append(child)
# new_model = nn.Sequential(*l)
# net = new_model
	
# load = torch.load('./checkpoint/gd1BestModel.t7')
# net.load_state_dict(load['net'])

# for param in net.parameters():
#  	param.requires_grad = True

if args.resume:
     # Load Best Model (Test Accuracy) from checkpoint.
     print('==> Resuming from checkpoint..')
     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
     load_dir ='./checkpoint/'+ args.name + 'BestModel.t7'
     checkpoint = torch.load(load_dir)
     net.load_state_dict(checkpoint['net'])
     best_acc = checkpoint['acc']
     start_epoch = checkpoint['epoch']

if args.resumebestloss:
    # Load Best Model (Train Loss) from checkpoint.
     print('==> Resuming from checkpoint..')
     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
     load_dir ='./checkpoint/'+ args.name + 'BestLossModel.t7'
     checkpoint = torch.load(load_dir)
     net.load_state_dict(checkpoint['net'])
     best_loss = checkpoint['best_loss']
     start_epoch = checkpoint['epoch']
     
#---------------------------------------Printing Network Stats and Moving to CUDA--------------------------------------
flops, params = get_model_complexity_info(net, (224, 224), as_strings=False, print_per_layer_stat=False)
print('==> Model Flops:{}'.format(flops))
print('==> Model Params:' + str(params))

net = net.to(device)
if device == 'cuda':
     #net = torch.nn.DataParallel(net, device_ids=[0,2,3,4])
     cudnn.benchmark = True
#-----------------------------------------------Optimizer----------------------------------------------
print('==>Setting Up Optimizer..')
criterion = nn.CrossEntropyLoss()
if args.optim=='adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
elif args.optim == 'rms':
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99, momentum=0.9, weight_decay = args.wd)
elif args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

step_size = 6*len(train_loader)#30000
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=step_size)

#--------------------------------------------

def train(epoch):
    global best_loss
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        scheduler.step()
        #optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    writer.add_scalar('TrainLoss', train_loss, epoch)
    writer.add_scalar('TrainAccuracy', (1.0*correct)/total, epoch)
    mlflow.log_metric('TrainLoss', train_loss, epoch)
    mlflow.log_metric('TrainAccuracy', (1.0*correct)/total, epoch)
    state = {
            'net': net.state_dict(),
            'best_loss': loss,
            'epoch': epoch,
        }
    if best_loss > loss:
        print('Saving Best Loss Model..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_dir = './checkpoint/'+ args.name + 'BestLossModel.t7'
        torch.save(state, save_dir)
        best_loss = loss

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    
    writer.add_scalar('TestLoss', test_loss, epoch)
    writer.add_scalar('TestAccuracy', (1.0*correct)/total, epoch)
    mlflow.log_metric('TestLoss', test_loss, epoch)
    mlflow.log_metric('TestAccuracy', (1.0*correct)/total, epoch) 

    acc = 100.*correct/total
    best_acc_int = int(best_acc*total/100.)
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
    if correct > best_acc_int:
        print('Saving Best Model..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_dir = './checkpoint/'+ args.name + 'BestModel.t7'
        torch.save(state, save_dir)
        best_acc = acc
    else:
        print('Saving Last Model..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_dir = './checkpoint/'+ args.name + 'LastEpoch.t7'
        torch.save(state, save_dir)

#-------------------------------------Main Part---------------------
with mlflow.start_run():
    mlflow.set_tag('Optim', args.optim)
    mlflow.set_tag('Name', args.name)
    mlflow.log_param('LR', str(args.lr))
    print('==> Started Training model..')
    for epoch in range(start_epoch, start_epoch+args.epoch):
        if args.optim =='rms' or args.optim =='sgd':
            if args.decay:
                lr = args.lr * (0.01 ** (epoch//10))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                mlflow.log_metric('Learning Rate', lr, epoch)
                writer.add_scalar('Learning Rate', lr, epoch)
        train(epoch)
        test(epoch)
        

writer.close()
#---------------------------------------

