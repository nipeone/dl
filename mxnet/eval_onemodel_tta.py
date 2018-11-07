#coding=utf-8

from __future__ import print_function

import argparse
from time import time
import mxnet as mx
import os
from mxnet.contrib import onnx as onnx_mxnet
from mxnet import gluon, nd
import numpy as np
import multiprocessing
from mxnet import gluon, nd, autograd
from mxnet.gluon.data.vision import transforms
import cPickle as pickle


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser(description='mxnet testing')
parser.add_argument('--path', '-p', type=str, help='the path of image need to test')
parser.add_argument('--model', '-m',default='model', type=str, help='the path of image need to test')
parser.add_argument('--count',default=1,type=int,help='evaluate count')
parser.add_argument('--epoch',default=0,type=int,help='epoch of exported model')
parser.add_argument('--seed', default=0, type=int, help='random seed')

args = parser.parse_args()


# Import the ONNX model into MXNet's symbolic interface
model_path= args.path
count = args.count
load_epoch=args.epoch
num_gpus=2
ctx = [mx.gpu(i) for i in range(num_gpus)]
#sym, arg, aux = onnx_mxnet.import_model(onnx_path)


structure=model_path+"-symbol.json"
parameter="%s-%04d.params"%(model_path,args.epoch)
print (structure)
print (parameter)
net = gluon.SymbolBlock.imports(structure,['data'],parameter,ctx=ctx)
#print("Loaded %s!"%(model_path))

#sym, arg, aux = mx.model.load_checkpoint(model_path, load_epoch)

#mod = mx.mod.Module(symbol=sym, data_names=data_names, context=ctx, label_names=None)
#mod = mx.mod.Module(symbol=sym,context=devs, label_names=None)
#mod.bind(for_training=False, data_shapes=[(data_names[0],(1,3,224,224))], label_shapes=None)
#mod.set_params(arg_params=arg, aux_params=aux, allow_missing=True, allow_extra=True)

#data_names = [graph_input for graph_input in sym.list_inputs()
#                      if graph_input not in arg and graph_input not in aux]
#print(data_names)

#net = gluon.nn.SymbolBlock(outputs=sym, inputs=mx.sym.var(data_names[0]))
#net.collect_params().initialize(ctx=ctx)
#net_params = net.collect_params()
#for param in arg:
#    if param in net_params:
#        net_params[param]._load_init(arg[param], ctx=ctx)
#for param in aux:
#    if param in net_params:
#        net_params[param]._load_init(aux[param], ctx=ctx)

net.hybridize()

BATCH_SIZE = 16
#NUM_WORKERS = multiprocessing.cpu_count()
NUM_WORKERS = 4


if args.seed != 0:
    mx.random.seed(args.seed)

data_dirs={"train":"/home/gfkd/bito8/train_dir","val":"/home/gfkd/bito8/validation_dir","test":"/home/gfkd/bito8/test_dir"}


def default_loader(path):
    try:
        img = mx.image.imread(path,to_rgb=1)
        return img
    except:
        print("Cannot read image: {}".format(path))


random_withflip_transforms = transforms.Compose([
        #mxnet v1.2 v1.3中Resize不一样
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor()
    ])

random_transforms = transforms.Compose([
        #mxnet v1.2 v1.3中Resize不一样
        transforms.RandomResizedCrop((224,224)),
        #transforms.RandomFlipLeftRight(),
        transforms.ToTensor()
    ])

center_transforms = transforms.Compose([
    transforms.Resize(256,True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ])

# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customDataset(gluon.data.Dataset):
    def __init__(self, img_path, txt_path, loader = default_loader):
        with open(os.path.join(img_path,txt_path)) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip('\n').rstrip().split()[0]) for line in lines]
            self.img_label = [int(line.strip('\n').rstrip().split()[1]) for line in lines]
        self._loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        img_name = self.img_name[index]
        label = self.img_label[index]
        img = self._loader(img_name)
        try:
            if count == 1:
                #resized_img = mx.image.resize_short(img,256,1)
                img = center_transforms(img)
            if count >1 and count <4:
                img = random_transforms(img)
            if count >3:
                img = random_withflip_transforms(img)
        except:
            print("Cannot transform image: {}".format(img_name))
        return img, label


testset = customDataset(img_path=data_dirs['test'],
                                txt_path='labels')

testloader = gluon.data.DataLoader(testset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
dataset_sizes = {x: len(testset) for x in ["test"]}
print("Test dataset: {} images".format(len(testset)))

def batch_fn(d,l, ctx):
    data = gluon.utils.split_and_load(d, ctx_list=ctx, batch_axis=0, even_split=False)
    label = gluon.utils.split_and_load(l, ctx_list=ctx, batch_axis=0, even_split=False)
    return data, label

def validate(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for _, batch in enumerate(val_data):
        data,label = batch_fn(batch[0],batch[1],ctx)
        outputs = [net(X.astype(np.float32,copy=False)) for X in data]
        metric.update(label, outputs)

    return metric.get()


def evaluate_accuracy_gluon(data_iterator, net):
   num_instance = nd.zeros(1, ctx=ctx[0])
   sum_metric = nd.zeros(1,ctx=ctx[0], dtype=np.int32)
   all_softlabels=[]
   all_labels=[]
   for data, label in data_iterator:
       all_labels.append(label.asnumpy())
       #outputs=[]
       #for i,data in enumerate(datas):
       #    data = data.astype(np.float32).as_in_context(mx.gpu(i))
       #    label = label.astype(np.int32).as_in_context(mx.gpu(i))
       #    output = net(data)
       #    outputs.append(output.expand_dims(axis=0))
       #output = nd.concatenate(outputs).mean(0)
       data = data.astype(np.float32).as_in_context(ctx[0])
       label = label.astype(np.int32).as_in_context(ctx[0])
       output = net(data)
       prediction = nd.argmax(output, axis=1).astype(np.int32)
       all_softlabels.append(output.asnumpy())
       num_instance += len(prediction)
       sum_metric += (prediction==label).sum()
   accuracy = (sum_metric.astype(np.float32)/num_instance.astype(np.float32))
   with open("pkl/eval-{}-tta{}-softlabels.pkl".format(args.model,args.count),'w') as f :
                pickle.dump(all_softlabels,f)
   with open("pkl/eval-{}-tta{}-labels.pkl".format(args.model,args.count),'w') as f :
                pickle.dump(all_labels,f)
   return accuracy.asscalar()

if __name__=='__main__':
    start=time()
    result = evaluate_accuracy_gluon(testloader,net)
    _,result1 = validate(net,testloader,ctx)
    print ('time: %f sec' %(time()-start))
    print (result)
    print (result1)
    #predict_single()
    #predict_multi()
    #img=mx.image.imread(args.path)
    #batch = transform(img)
    #result=run_batch(net,[batch])
    #print (result)