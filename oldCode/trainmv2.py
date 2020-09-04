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
parser.add_argument('--change', nargs='+', help='Layers to be Modified', default=[])
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resumebestloss', action='store_true', help='resume from best loss checkpoint')
parser.add_argument('--name', type=str, help='TensorboardName and Name to save the models', required=True)
parser.add_argument('--epoch1', type=int, help='Number of Epochs to Train Freezed', required=True)
parser.add_argument('--epoch2', type=int, help='Number of Epochs to Train Unfreezed', required=True)
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
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv2 import InvertedResidualFriendly

net = MobileNetV2()
state_dict = torch.load('./pretrainedmodels/mobilenet_v2.pth')
net.load_state_dict(state_dict, strict=True)
cfg = [
    #inp, out ,stride, expansion
    [32, 16, 1, 1],

    [16, 24, 2, 6],
    [24, 24, 1, 6],

    [24, 32, 2, 6],
    [32, 32, 1, 6],
    [32, 32, 1, 6],

    [32, 64, 2, 6],
    [64, 64, 1, 6],
    [64, 64, 1, 6],
    [64, 64, 1, 6],

    [64, 96, 1, 6],
    [96, 96, 1, 6],
    [96, 96, 1, 6],

    [96, 160, 2, 6],
    [160, 160, 1, 6],
    [160, 160, 1, 6],

    [160, 320, 1, 6],
    ]

for x in args.change:
    i = int(x)
    net.features[i] = InvertedResidualFriendly(*cfg[i-1])

for param in net.parameters():
  	param.requires_grad = False

for x in args.change:
    i = int(x)
    for param in net.features[i].parameters():
  	    param.requires_grad = True



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
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=4e-5)
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
        # scheduler.step()
        optimizer.step()

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
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
    if acc > best_acc:
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
    mlflow.set_tag('Name', args.name)
    print('==> Started Training model..')
    for epoch in range(start_epoch, start_epoch+args.epoch1):
        train(epoch)
        test(epoch)
    
    for param in net.parameters():
  	    param.requires_grad = True
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=4e-5)

    for epoch in range(args.epoch1, args.epoch1+args.epoch2):
        if args.decay:
            lr = 1e-3 * (0.01 ** (epoch//10))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            mlflow.log_metric('Learning Rate', lr, epoch)
            writer.add_scalar('Learning Rate', lr, epoch)
        train(epoch)
        test(epoch)

writer.close()
#---------------------------------------

