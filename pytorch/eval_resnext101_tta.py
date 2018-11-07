#coding=utf-8
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms

import time
import os,argparse
import copy
from torch.utils.data import Dataset
import tta_wrapper as tta
import cPickle as pickle

import resnext_101_64x4d

kind_dataset=['test']
data_dirs={"test":"/home/gfkd/bito8/test_dir"}

from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default="resnext101", type=str, help='model name')
parser.add_argument('--count', default=1, type=int, help='model name')
parser.add_argument('--gpu', default="0", type=str, help='gpu index')
args = parser.parse_args()

gpus=args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
gpu_nums=len(gpus.split(','))

# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


center=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

withAffine = transforms.Compose([
        transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(10.0),
        transforms.ToTensor(),
    ])
withoutAffine = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomAffine(10.0),
        transforms.ToTensor(),
    ])

# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, img_path, txt_path , data_transform=None, loader = default_loader):
        with open(os.path.join(img_path,txt_path)) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip('\n').rstrip().split()[0]) for line in lines]
            self.img_label = [int(line.strip('\n').rstrip().split()[1]) for line in lines]
        self.data_transform = data_transform
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        img_name = self.img_name[index]
        label = self.img_label[index]
        img = self.loader(img_name)
        #arr=[]
        count = args.count
        if self.data_transform is not None:
            try:
                if count == 1:
                    img = center(img)
                if count>1 and count <4:
                    img = withAffine(img)
                if count>3:
                    img = withoutAffine(img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label

data_transforms = {
    'test': transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(10.0),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

use_gpu = torch.cuda.is_available()
batch_size =32
num_class = 8
image_datasets = {x: customData(img_path=data_dirs[x],
                                txt_path='labels',
                                data_transform=data_transforms[x]) for x in kind_dataset}

# wrap your data and label into Tensor
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=4) for x in kind_dataset}

dataset_sizes = {x: len(image_datasets[x]) for x in kind_dataset}



def eval_model(model,  num_epochs):
    since = time.time()

    #best_model_wts = model.state_dict()
    # best_model_wts = copy.deepcopy(model.state_dict())
    #best_acc = 0.0

    for epoch in range(num_epochs):
        begin_time = time.time()
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['test']:

            # Set model to evaluate mode
            model.train(False)
            #for model in models: 
            # running_loss = 0.0
            running_corrects = 0.
            f1="pkl/eval-{}-tta{}-softlabels.pkl".format(args.model,args.count)
            f2='pkl/eval-{}-tta{}-labels.pkl'.format(args.model,args.count)
            # Iterate over data.
            #outputs_tta_result=[]
            all_labels=[]
            all_softlabels=[]
            for idx,data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = data
                all_labels.append(labels.clone().data.cpu().numpy())
                if use_gpu:
                    inputs,labels=Variable(inputs.cuda()),Variable(labels.cuda())
                else:
                    inputs,labels = Variable(inputs),Variable(labels)
              
                #for inputs in inputs_list:
                #    # wrap them in Variable
                #    if use_gpu:
                #        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                #    else:
                #        inputs, labels = Variable(inputs), Variable(labels)
                #    # forward compute
                #    output=model(inputs)
                #    torch.cuda.empty_cache()
                #    outputs.append(output)

                #outputs = torch.stack(outputs,0).mean(0)
                #outputs_tta_result.append(outputs.data.cpu().numpy())

                #outputs=[]
                #for model in models:
                #   outputs.append(model(inputs))
                #outputs = [model(inputs) for model in models]
                # _, preds = torch.max(outputs.data, 1)
                #head = outputs[0]             
                #for output in outputs[1:]:
                #    head+=output
                outputs=model(inputs)
                all_softlabels.append(outputs.data.cpu().numpy())
                _, preds = torch.max(outputs, 1)
                #_, preds = torch.max(outputs, 1)
                # loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                # if phase == 'train':
                #     loss.backward()
                #     optimizer.step()

                # statistics
                #running_loss += loss.data[0]
                # running_loss += loss.data[0] + inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
                # running_loss += loss.item() * inputs.size(0)
                running_corrects += preds.eq(labels.data).cpu().sum().float()
                torch.cuda.empty_cache()
                #running_corrects += torch.sum(preds == labels.data).float()

                # print result every 10 batch
                if idx%10 == 0:
                #     batch_loss = running_loss / (batch_size*count_batch)
                    batch_acc = running_corrects / (batch_size*(idx+1))
                    print('{} Epoch [{}] Batch [{}] Acc: {:.4f} Time: {:.4f}s'. \
                           format(phase, epoch, idx, batch_acc, time.time()-begin_time))
                    begin_time = time.time()

            # epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{}  Acc: {:.4f}'.format(phase, epoch_acc))

            # save model
            with open(f1,'w') as f :
                pickle.dump(all_softlabels,f)
            with open(f2,'w') as f :
                pickle.dump(all_labels,f)

        print()

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))



if __name__ == '__main__':

    # get model and replace the original fc layer with your fc layer
    # model_ft = models.resnet101(pretrained=True)
    #model_ft = torch.load("output_bak/best_resnet101.pkl")
    checkpoint = torch.load("./checkpoint/ckpt.t70_20181018")
    model_ft = resnext_101_64x4d.resnext_101_64x4d
    model_ft.load_state_dict(torch.load('resnext_101_64x4d.pth'))
    model_ft[-1][1]=nn.Linear(2048,num_class)
    model_state_orig = model_ft.state_dict()
    model_state = checkpoint['net']
    for item in model_state_orig.keys():
        model_state_orig[item] = model_state['module.'+item]

    best_acc=checkpoint['acc']
    model_ft.load_state_dict(model_state_orig)
    print ("best acc is " + str(best_acc))
    
       
    # for param in model_ft.parameters():
    #     param.requires_grad = False
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, num_class)

    # if use gpu
    if use_gpu:
        #for idx,model in enumerate(models):
        #    models[idx]= model.cuda()
        model_ft = model_ft.cuda()

    # define cost function
    # criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.005, momentum=0.9)

    # Decay LR by a factor of 0.2 every 5 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.2)

    # multi-GPU
    model_ft = torch.nn.DataParallel(model_ft, device_ids=range(gpu_nums))
    #for idx,model in enumerate(models):
    #    models[idx] = torch.nn.DataParallel(model,device_ids=[0])

    #wrapper = tta.TTA_ModelWrapper(model_ft)
    # train model
    eval_model(model=model_ft,num_epochs=1)