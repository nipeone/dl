#coding=utf-8

from __future__ import print_function

import os,logging,math,time
import argparse
import mxnet as mx
from mxnet import gluon,nd
import numpy as np
from mxnet.gluon.data.vision import  transforms
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data import Dataset
import mxnet.autograd as autograd

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser(description='Mxnet digest Training')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="resnext101", type=str,
                    help='model type (default: resnet101)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batchsize', default=64, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--mixup_alpha', default=0.2, type=float,
                    help='mixup interpolation coefficient (default: 1)')
args = parser.parse_args()

filehandler = logging.FileHandler('log/train_resnext101.log')
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(args)

BATCH_SIZE=args.batchsize
NUM_CLASSES=8
NUM_WORKERS=4

if args.seed != 0:
    mx.random.seed(args.seed)

data_dirs={"train":"/home/gfkd/bito8/train_dir","val":"/home/gfkd/bito8/validation_dir"}


def default_loader(path):
    try:
        img=mx.image.imread(path,to_rgb=1)
        return img
    except:
        print("Cannot read image: {}".format(path))

data_transforms= {
        "train":transforms.Compose([
        #mxnet v1.2 v1.3中Resize不一样
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor()
            ]),
        "val":transforms.Compose([
        transforms.CenterCrop((224,224)),
        transforms.ToTensor()
        
           ])}

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
                if self._mode != 'train':
                    img = mx.image.resize_short(img,256,1)
                img = self._data_transform(img)
            except Exception,e:
                print (e.message)
                print("Cannot transform image: {}".format(img_name))
        return img, label

###############################################################
#define train data and validation data
###############################################################
data_sets = {x:customData(img_path=data_dirs[x],txt_path="labels",mode=x,data_transform=data_transforms[x]) for x in ["train","val"]}

train_loader = gluon.data.DataLoader(data_sets["train"], batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,last_batch='discard')

val_loader = gluon.data.DataLoader(data_sets["val"], batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,last_batch='discard')
dataset_sizes = {x: len(data_sets[x]) for x in ["train","val"]}


sym, arg_params, aux_params = mx.model.load_checkpoint('resnext-101-64x4d', 0)

print("Loaded checkpoint!")

data_names = [graph_input for graph_input in sym.list_inputs()
                      if graph_input not in arg_params and graph_input not in aux_params]
print(data_names)

ctx=[mx.gpu(i) for i in range(4)]

pre_trained = gluon.nn.SymbolBlock(outputs=sym, inputs=mx.sym.var(data_names[0]))
pre_trained.collect_params().initialize(ctx=ctx)
net_params = pre_trained.collect_params()
for param in arg_params:
    if param in net_params:
        net_params[param]._load_init(arg_params[param], ctx=ctx)
for param in aux_params:
    if param in net_params:
        net_params[param]._load_init(aux_params[param], ctx=ctx)
#修改最后一层
dense_layer=gluon.nn.Dense(NUM_CLASSES)
dense_layer.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

net = gluon.nn.HybridSequential()
net.add(pre_trained)
net.add(dense_layer)

LEARNING_RATE = 0.005
WDECAY = 0.0001
MOMENTUM = 0.9


best_acc=0.0

#softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

net.hybridize()


def batch_fn(d,l, ctx):
    data = gluon.utils.split_and_load(d, ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(l, ctx_list=ctx, batch_axis=0)
    return data, label

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
train_metric = mx.metric.RMSE()

def evaluate(model, data_iterator,ctx):
    acc_top1.reset()
    acc_top5.reset()
    for i, (data,label) in enumerate(data_iterator):
        data, label = batch_fn(data,label, ctx)
        outputs = [net(X.astype(np.float32, copy=False)) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)

#def evaluate(model,data_iterator):
#   num_instance = nd.zeros(1, ctx=ctx)
#   sum_metric = nd.zeros(1,ctx=ctx, dtype=np.int32)
#   for i, (data, label) in enumerate(data_iterator):
#       data = data.astype(np.float32).as_in_context(ctx)
#       label = label.astype(np.int32).as_in_context(ctx)
#       #data, label = batch_fn(data,label, ctx)
#       output = model(data)
#       prediction = nd.argmax(output, axis=1).astype(np.int32)
#       num_instance += len(prediction)
#       sum_metric += (prediction==label).sum()
#   accuracy = (sum_metric.astype(np.float32)/num_instance.astype(np.float32))
#   return accuracy.asscalar()


def mixup_data(data,label,classes=NUM_CLASSES):
    if isinstance(data,nd.NDArray):
        data = [data]
    if args.mixup_alpha > 0:
        lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
    else:
        lam = 1.
    data = [lam*X + (1-lam)*X[::-1] for X in data]
    if isinstance(label,nd.NDArray):
        label=[label]
    res = []
    for l in label:
        y1 = l.one_hot(classes, on_value = 1., off_value = 0.)
        y2 = l[::-1].one_hot(classes, on_value = 1., off_value = 0.)
        res.append(lam*y1 + (1-lam)*y2) 
    return data,res


def adjust_learning_rate(epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 20:
        lr /= 10
    if epoch >= 40:
        lr /= 10
    if epoch >= 60:
        lr /= 10
    if epoch >= 80:
        lr /= 10
    if epoch >= 100:
        lr /= 10
    return lr

# Train a given model using MNIST data
def train(model,epoch,loss_fn,ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    best_val_score = 1.0
    # Use Adam optimizer
    trainer = gluon.Trainer(model.collect_params(), 'sgd', 
                           {'learning_rate': args.lr,
                           'wd':WDECAY,
                           'momentum':MOMENTUM})

    since=time.time() # trainning start time

    # Train for one epoch
    for i in range(epoch):
        begin=time.time()
        train_metric.reset()
        # Iterate through the images and labels in the training data
        for batch_idx, (data, label) in enumerate(train_loader):
            # get the images and labels
            #data = data.as_in_context(ctx)
            #label = label.as_in_context(ctx)
            data, label = batch_fn(data,label, ctx)
            data,label = mixup_data(data,label)
            # Ask autograd to record the forward pass
            with autograd.record():
                # Run the forward pass
                #output = model(data)
                # Compute the loss
                #loss = loss_fn(output, label)
                # Run the forward pass
                outputs = [model(X.astype(np.float32, copy=False)) for X in data]
                # Compute the loss
                loss = [loss_fn(yhat, y.astype(np.float32, copy=False)) for yhat, y in zip(outputs, label)] # Compute gradients
            #loss.backward()
            for l in loss:
                l.backward()
            # Update parameters
            trainer.step(BATCH_SIZE)
            trainer.set_learning_rate(adjust_learning_rate(i))
            
            output_softmax = [mx.nd.SoftmaxActivation(out.astype('float32', copy=False)) for out in outputs]
            train_metric.update(label,outputs)
            # Print loss once in a while
            if batch_idx%10==0 and batch_idx >0:
                print('Epoch :[{0}]; Batch [{1}] loss: {2:.4f}'.format(i,batch_idx, nd.concatenate(loss).mean(0).asscalar()))
        #nd.waitall()
        train_metric_name, train_metric_score = train_metric.get()
        err_top1_val,err_top5_val=evaluate(model,val_loader,ctx)
        print('Epoch [{0}] cost {1:.0f}s '.format(i,time.time()-begin))         
        print("Epoch [{0}] Test Accuracy {1:.4f} ".format(i, 1-err_top1_val))
        logger.info('[Epoch %d] training: %s=%f'%(i, train_metric_name, train_metric_score))
        logger.info('[Epoch %d] validation: err-top1=%f err-top5=%f'%(i, err_top1_val, err_top5_val))

        if err_top1_val < best_val_score: 
            checkpoint(model,i)
            best_val_score = err_top1_val

    #train finished time
    time_elapsed=time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def checkpoint(net,epoch):
    net.export('pdl-resnext',epoch=epoch)

def main():
    print ("start trainning")
    train(net,args.epoch,softmax_cross_entropy,ctx)
    print ("finish trainning")

if __name__ == '__main__':
    main()