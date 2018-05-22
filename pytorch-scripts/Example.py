#Demo for CS7-GV1

#general modules
from __future__ import print_function, division
import os
import argparse
import time
import copy
import numpy as np
import random
from sklearn.metrics import confusion_matrix
#pytorch modules
import torch
import torch.nn as nn
from torchvision import datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.autograd import Variable
import pdb
import itertools

#user defined modules
import Augmentation as ag
import Models
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from finetune import FineTuneModel
from Test import Test
parser = argparse.ArgumentParser(description='CS7-GV1 Final Project');

writer = SummaryWriter()

#add/remove arguments as required. It is useful when tuning hyperparameters from bash scripts
parser.add_argument('--aug', type=str, default = '', help='data augmentation strategy')
parser.add_argument('--datapath', type=str, default='', 
               help='root folder for data.It contains two sub-directories train and val')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')               
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model')
parser.add_argument('--batch_size', type=int, default = 128,
                    help='batch size')
parser.add_argument('--model', type=str, default = None, help='Specify model to use for training.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument('--tag', type=str, default=None,
                    help='unique_identifier used to save results')
args = parser.parse_args();
if not args.tag:
    print('Please specify tag...')
    exit()
print (args)

#Define augmentation strategy
augmentation_strategy = ag.Augmentation(args.aug);
data_transforms = augmentation_strategy.applyTransforms();
##

#Root directory
data_dir = args.datapath;
##

######### Data Loader ###########
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
             for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,
                                               shuffle=True, num_workers=16) # set num_workers higher for more cores and faster data loading
             for x in ['train', 'val']}
                 
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
# import ipdb; ipdb.set_trace()
dset_classes = dsets['train'].classes
#################################

#set GPU flag
use_gpu = args.cuda;
##

#Load model . Once you define your own model in Models.py, you can call it from here. 
if args.model == "ResNet18":
    current_model = Models.resnet18(args.pretrained)
    # num_ftrs = current_model.fc.in_features
    # current_model.fc = nn.Linear(num_ftrs, len(dset_classes));
    current_model = FineTuneModel(current_model, args.model, len(dset_classes), args.pretrained)
elif args.model == 'AlexNet':
    current_model = Models.alexnet(args.pretrained)
    current_model =  FineTuneModel(current_model, args.model, len(dset_classes), args.pretrained)
elif args.model == 'VGG16':
    current_model = Models.vgg16(args.pretrained)
    current_model = FineTuneModel(current_model, args.model, len(dset_classes), args.pretrained)
elif args.model == 'Demo':
    current_model = Models.demo_model();
elif args.model == 'SSNet':
    current_model = Models.ssnet_model();
else :
    print ("Model %s not found"%(args.model))
    exit();    


if use_gpu:
    current_model = current_model.cuda();
    
# uses a cross entropy loss as the loss function
# http://pytorch.org/docs/master/nn.html#
criterion = nn.CrossEntropyLoss()

#uses stochastic gradient descent for learning
# http://pytorch.org/docs/master/optim.html
optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, current_model.parameters()), lr=args.lr, momentum=0.9, weight_decay=0.003)

#the learning rate condition. The ReduceLROnPlateau class reduces the learning rate by 'factor' after 'patience' epochs.
scheduler_ft = ReduceLROnPlateau(optimizer_ft, 'min', factor = 0.5,patience = 3, verbose = True)

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# def randomTrainingExample():
#     category = randomChoice(dset_classes)
#     line = randomChoice(category_lines[category])
#     category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
#     line_tensor = Variable(lineToTensor(line))
#     return category, line, category_tensor, line_tensor


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    # import ipdb;
    # ipdb.set_trace()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0
    epoch = 0
    # for epoch in range(num_epochs):
    while(optimizer.param_groups[0]['lr']>0.00001):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            top1_accu=top5_accu=0.0
            # Iterate over data.
            for count, data in enumerate(dset_loaders[phase]):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                if count%10 == 0:
                    print('Batch %d || Running Loss = %0.6f || Running Accuracy = %0.6f'%(count+1,running_loss/(args.batch_size*(count+1)),running_corrects/(args.batch_size*(count+1))))
                    print('Running Loss = %0.6f'%(running_loss/(args.batch_size*(count+1))))

                top1_running_accu, top5_running_accu = accuracy(outputs,labels,(1,5))
                top1_accu += top1_running_accu
                top5_accu += top5_running_accu

                # top1_accu = top1_accu/(count+1)
                # top5_accu = top5_accu/(count+1)
                if count%10 == 0:
                    print('Printing top1 and top5 accu:')
                    print('Batch %d || Running Loss = %0.6f || top1 Accuracy = %0.6f || top5 Accuracy = %0.6f '%(count+1,running_loss/(args.batch_size*(count+1)),top1_accu/(count+1),top5_accu/(count+1)))
                    # print('Batch %d || Running Loss = %0.6f || top1 Accuracy = %0.6f || top5 Accuracy = %0.6f ' % (count + 1, running_loss / (args.batch_size * (count + 1)),
                    # top1_running_accu , top5_running_accu ))

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            # import ipdb;
            # ipdb.set_trace()
            epoch_top1_accuracy = top1_accu / len(dset_loaders[phase])
            epoch_top5_accuracy = top5_accu / len(dset_loaders[phase])

            writer.add_scalar("per epoch/learning rate", optimizer.param_groups[0]['lr'], epoch)

            if phase=='train':
                writer.add_scalar("per epoch/training/loss",epoch_loss,epoch)
                # writer.add_scalar("per epoch/org_accuracy",epoch_acc,epoch)
                writer.add_scalar("per epoch/training/top1_accuracy", epoch_top1_accuracy, epoch)
                writer.add_scalar("per epoch/training/top5_accuracy", epoch_top5_accuracy, epoch)
            elif phase=='val':
                writer.add_scalar("per epoch/validation/loss", epoch_loss, epoch)
                # writer.add_scalar("per epoch/org_accuracy",epoch_acc,epoch)
                writer.add_scalar("per epoch/validation/top1_accuracy", epoch_top1_accuracy, epoch)
                writer.add_scalar("per epoch/validation/top5_accuracy", epoch_top5_accuracy, epoch)
            else:
                print('Phase not defined')

            # import ipdb; ipdb.set_trace()
            # for param in model.parameters():
            #     writer.add_histogram(param.__class__.__name__, param.data)

            print('Epoch %d || %s Loss: %.4f || Top1_Acc: %.4f || Top5_Acc: %.4f'%(epoch,
                phase, epoch_loss, epoch_top1_accuracy, epoch_top5_accuracy ) ,end = ' || ')
            #pdb.set_trace();
            if phase == 'val':
                print ('\n', end='');
                lr_scheduler.step(epoch_loss);

            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
        epoch+=1

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model



######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dset_loaders['val']):
        # import ipdb; ipdb.set_trace()
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dset_classes[preds[j]])+'\ncorrect label: {}'.format(dset_classes[labels.data[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

def confusion_matrix_plot(model, num_images=10):
        images_so_far = 0
        plt.figure()
        label_list = []
        pred_list = []
        scores = torch.cuda.FloatTensor()
        scores_list=[]
        for i, data in enumerate(dset_loaders['val']):
            inputs, labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            scores = torch.cat((scores, torch.nn.functional.softmax(outputs).data),0)

            for j in range(inputs.size()[0]):
                # images_so_far += 1
                pred_list.append(dset_classes[preds[j]])
                label_list.append(dset_classes[labels.data[j]])
                scores_list.append(float("{0:.5f}".format(scores.cpu().numpy()[i][j])))
                # if images_so_far == num_images:
                #     break
            # for j in range(preds):
            #     if labels.data[j] not in label_list:
            #         label_list.append(labels.data[j])
            

        label_list_number=[0.1 for label in label_list]
        pred_list_number=[1*scores_list[i] for i,pred in enumerate(pred_list)]
        fmt = '.2f'
        # import ipdb; ipdb.set_trace()
        cm = confusion_matrix(pred_list,label_list)[:20,:20]
        pred_list = sorted(list(set(pred_list)))[:20]
        label_list=pred_list
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > 0.5 else "black")
        plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(label_list))
        plt.xticks(tick_marks, pred_list, rotation=45)
        plt.yticks(tick_marks, label_list)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted Label')
        plt.show()


#comment the block below if you are not training 
######################

# trained_model = train_model(current_model, criterion, optimizer_ft, scheduler_ft,
#                      num_epochs=args.epochs);
# with open(args.tag+'.model', 'wb') as f:
#     torch.save(trained_model, f);
# writer.export_scalars_to_json("./all_scalars.json")
# writer.close()
# visualize_model(trained_model)
# plt.show()

##########################
#plotting confusion matrix
trained_model = torch.load(args.tag+'.model')
cm=confusion_matrix_plot(trained_model)




######################    
## uncomment the lines blow while testing.
#trained_model = torch.load(args.tag+'.model');
#testDataPath = 'data/test/'
#t = Test(args.aug,trained_model);
#scores = t.testfromdir(testDataPath);
#pdb.set_trace();
#np.savetxt(args.tag+'.txt', scores, fmt='%0.5f',delimiter=',')

#import csv
#with open('ssnet_incept.txt',newline='') as f:
#    r=csv.reader(f)
#    data=[line for line in r]
#with open('ssnet_incept_headers.txt','w',newline='') as f:
#    w=csv.writer(f)
#    row_headers=dset_classes
#    row_headers.insert(0,'imid')
#    w.writerow(row_headers)
#    for imid,line in enumerate(data):
#        line.insert(0,imid)
#        w.writerow(line)

