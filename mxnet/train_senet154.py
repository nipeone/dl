#coding=utf-8

################################################################################
# Hyperparameters
# ----------
#
# First, let's import all other necessary libraries.

import mxnet as mx
import numpy as np
import os, time, shutil,logging,argparse

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data import Dataset
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler

################################################################################
# We set the hyperparameters as following:

parser = argparse.ArgumentParser(description='Mxnet digest Training')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="resnext101", type=str,
                    help='model type (default: resnet101)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--gpu', default='0', type=str, help='gpu used')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batchsize', default=64, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--mixup_alpha', default=0.2, type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--ckpt',default="ckpt",type=str,help='checkpoint path')
args = parser.parse_args()

filehandler = logging.FileHandler('log/train_senet154.log')
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(args)


################################################################################
# We set the hyperparameters as following:

gpus=args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

# 分类
classes = 8

ckpt_path=args.ckpt
epochs = args.epoch
lr = args.lr
per_device_batch_size = args.batchsize
momentum = 0.9
wd = 0.0001

lr_factor = 0.1
lr_steps = [20, 40, 60, np.inf]

num_gpus = len(gpus.split(","))
num_workers = 8
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = per_device_batch_size * max(num_gpus, 1)

data_dirs={"train":"/home/gfkd/bito8/train_dir","val":"/home/gfkd/bito8/validation_dir","test":"/home/gfkd/bito8/test_dir"}

################################################################################
# Things to keep in mind:
#
# 1. ``epochs = 5`` is just for this tutorial with the tiny dataset. please change it to a larger number in your experiments, for instance 40.
# 2. ``per_device_batch_size`` is also set to a small number. In your experiments you can try larger number like 64.
# 3. remember to tune ``num_gpus`` and ``num_workers`` according to your machine.
# 4. A pre-trained model is already in a pretty good status. So we can start with a small ``lr``.
#
# Data Augmentation
# -----------------
#
# In transfer learning, data augmentation can also help.
# We use the following augmentation in training:
#
# 2. Randomly crop the image and resize it to 224x224
# 3. Randomly flip the image horizontally
# 4. Randomly jitter color and add noise
# 5. Transpose the data from height*width*num_channels to num_channels*height*width, and map values from [0, 255] to [0, 1]
# 6. Normalize with the mean and standard deviation from the ImageNet dataset.
#
jitter_param = 0.4
lighting_param = 0.1

data_transforms= {
        "train":transforms.Compose([
            #mxnet v1.2 v1.3中Resize不一样
            transforms.RandomResizedCrop((224,224)),
            transforms.RandomFlipLeftRight(),
            transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                    saturation=jitter_param),
            transforms.RandomLighting(lighting_param),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        "val":transforms.Compose([
            transforms.Resize(255,True),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        "test":transforms.Compose([
            transforms.Resize(255,True),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

################################################################################
# With the data augmentation functions, we can define our data loaders:

def default_loader(path):
    try:
        img=mx.image.imread(path,to_rgb=1)
        return img
    except:
        print("Cannot read image: {}".format(path))

# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, img_path, txt_path,mode='', data_transform=None, loader = default_loader):
        with open(os.path.join(img_path,txt_path)) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip('\n').rstrip().split()[0]) for line in lines]
            #self.img_label = [labels_map[line.strip('\n').rstrip().split()[1]] for line in lines]
            self.img_label = [int(line.strip('\n').rstrip().split()[1]) for line in lines]
        self._data_transform = data_transform
        self._loader = loader
        self._mode = mode

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        img_name = self.img_name[index]
        label = self.img_label[index]
        img = self._loader(img_name)

        if self._data_transform is not None:
            try:
                img = self._data_transform(img)
            except Exception,e:
                print (e.message)
                print("Cannot transform image: {}".format(img_name))
        return img, label


data_sets = {x:customData(img_path=data_dirs[x],txt_path="labels",mode=x,data_transform=data_transforms[x]) for x in ["train","val","test"]}

train_loader = gluon.data.DataLoader(data_sets["train"], batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,last_batch='discard')

val_loader = gluon.data.DataLoader(data_sets["val"], batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,last_batch='discard')

test_loader = gluon.data.DataLoader(data_sets["test"], batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,last_batch='discard')

dataset_sizes = {x: len(data_sets[x]) for x in ["train","val","test"]}

################################################################################
#
# Note that only ``train_data`` uses ``transform_train``, while
# ``val_data`` and ``test_data`` use ``transform_test`` to produce deterministic
# results for evaluation.
#
# Model and Trainer
# -----------------
#
# We use a pre-trained ``ResNet50_v2`` model, which has balanced accuracy and
# computation cost.

model_name = 'senet_154'
finetune_net = get_model(model_name, pretrained=True)
with finetune_net.name_scope():
    finetune_net.output = nn.Dense(classes)
finetune_net.output.initialize(init.Xavier(), ctx = ctx)
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()


def batch_fn(d,l, ctx):
    data = gluon.utils.split_and_load(d, ctx_list=ctx, batch_axis=0, even_split=False)
    label = gluon.utils.split_and_load(l, ctx_list=ctx, batch_axis=0, even_split=False)
    return data, label

################################################################################
# Here's an illustration of the pre-trained model
# and our newly defined model:
#
# |image-model|
#
# Specifically, we define the new model by::
#
# 1. load the pre-trained model
# 2. re-define the output layer for the new task
# 3. train the network
#
# This is called "fine-tuning", i.e. we have a model trained on another task,
# and we would like to tune it for the dataset we have in hand.
#
# We define a evaluation function for validation and testing.

def validate(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for _, batch in enumerate(val_data):
        data,label = batch_fn(batch[0],batch[1],ctx)
        outputs = [net(X.astype(np.float32,copy=False)) for X in data]
        metric.update(label, outputs)

    return metric.get()

################################################################################
# Training Loop
# -------------
#
# Following is the main training loop. It is the same as the loop in
# `CIFAR10 <dive_deep_cifar10.html>`__
# and ImageNet.
#
# .. note::
#
#     Once again, in order to go through the tutorial faster, we are training on a small
#     subset of the original ``MINC-2500`` dataset, and for only 5 epochs. By training on the
#     full dataset with 40 epochs, it is expected to get accuracy around 80% on test data.

def train(net,train_data,epochs,ctx):
    lr_counter = 0
    num_batch = len(train_data)
    best_val_acc = 0.0
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
                        'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        if epoch == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate*lr_factor)
            lr_counter += 1

        tic = time.time()
        train_loss = 0
        metric.reset()

        for _, batch in enumerate(train_data):
            data,label = batch_fn(batch[0],batch[1],ctx)
            with ag.record():
                outputs = [net(X.astype(np.float32,copy=False)) for X in data]
                loss = [L(yhat, y.astype(np.float32,copy=False)) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

            metric.update(label, outputs)
        mx.nd.waitall()
        _, train_acc = metric.get()
        train_loss /= num_batch

        _, val_acc = validate(net, val_loader, ctx)

        print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1fs' %
                (epoch, train_acc, train_loss, val_acc, time.time() - tic))
        logger.info('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1fs' %
                        (epoch, train_acc, train_loss, val_acc, time.time() - tic))

        #checkpoint
        if val_acc > best_val_acc:
            checkpoint(net,epoch)
            best_val_acc = val_acc
    print('[Finished] best Val-acc: %.3f' % (best_val_acc))
    logger.info('[Finished] best Val-acc: %.3f' % (best_val_acc))
    _, test_acc = validate(net, test_loader, ctx)
    print('[Finished] Test-acc: %.3f' % (test_acc))
    logger.info('[Finished] Test-acc: %.3f' % (test_acc))

def checkpoint(net,epoch):
    net.export(os.path.join(ckpt_path,'pdl-senext'),epoch=epoch)

def main():
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    train(finetune_net,train_loader,epochs,ctx)

if __name__ == '__main__':
    main()