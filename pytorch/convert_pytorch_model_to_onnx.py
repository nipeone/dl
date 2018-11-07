#coding=utf-8

import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.onnx
from torchvision import  models, transforms
from torch.jit import _unique_state_dict
from PIL import Image
import resnext_101_64x4d

#densenet169
#my_wgt=mymodel.state_dict()

#model = torchvision.models.densenet169(pretrained=True).cuda()
#model.load_state_dict(my_wgt)

num_class=8

def cvt_my_model(path):
    
    filepath,filename = os.path.split(path) 
    print (filepath)
    print (filename)
    ckpt=torch.load(path)
    net=ckpt['net'] 
    model=recursion_original_module(net)

    model_ft=model.cuda()
    model_ft.train(False)

    dummy_input = Variable(torch.randn(1, 3, 224, 224)).cuda()
    #graph, torch_out =  _trace_and_get_graph_from_model(model_ft,dummy_input,False)
    #orig_state_dict_keys = _unique_state_dict(model).keys()
    #print(len(orig_state_dict_keys))
    #state_dict_keys = model.state_dict().keys()
    #print (len(state_dict_keys))
    #print (state_dict_keys)
     
    img = Image.open('test_dir/6_xr_200062513.jpg')
    img = img.convert('RGB')
    toTensor= transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    t = toTensor(img)

    input=Variable(torch.unsqueeze(t,0)).cuda()
    print (input.shape)
    torch_out = torch.onnx._export(model_ft, input, "./cvtmodels/{}.onnx".format(filename),export_params=True, verbose=True)
    print (torch_out)

def cvt_resnext(path):
    filepath,filename = os.path.split(path)
    ckpt = torch.load(path)
    net = resnext_101_64x4d.resnext_101_64x4d
    net.load_state_dict(torch.load('resnext_101_64x4d.pth'))
    print (net[-1][1])
    net[-1][1]=nn.Linear(2048,num_class)
    print (net[-1][1])
    model_state_orig = net.state_dict()
    model_state = ckpt['net']
    for item in model_state_orig.keys():
        model_state_orig[item] = model_state['module.'+item]
    net.load_state_dict(model_state_orig)
    model=recursion_original_module(net)
    model_ft = model.cuda()
    model_ft.train(False)
    
    dummy_input = Variable(torch.randn(1, 3, 224, 224)).cuda()
    
    img = Image.open('test_dir/6_xr_200062513.jpg')
    img = img.convert('RGB')
    toTensor= transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    t = toTensor(img)

    input = Variable(torch.unsqueeze(t,0)).cuda()
    print (input.shape)
    torch_out = torch.onnx._export(model_ft, input, "./cvtmodels/{}.onnx".format(filename),export_params=True, verbose=True)
    print (torch_out) 

def recursion_original_module(model):
    if isinstance(model,nn.parallel.DataParallel):
        return recursion_original_module(model.module)
    else:
        return model

def _trace_and_get_graph_from_model(model, args, training):

    # A basic sanity check: make sure the state_dict keys are the same
    # before and after running the model.  Fail fast!
    orig_state_dict_keys = _unique_state_dict(model).keys()

    # By default, training=False, which is good because running a model in
    # training mode could result in internal buffers getting updated, dropout
    # getting applied, etc.  If you really know what you're doing, you
    # can turn training=True (or None, to preserve whatever the original
    # training mode was.)
    #with set_training(model, training):
    trace, torch_out = torch.jit.get_trace_graph(model, args)

    if orig_state_dict_keys != _unique_state_dict(model).keys():
        raise RuntimeError("state_dict changed after running the tracer; "
                           "something weird is happening in your model!")

    return trace.graph(), torch_out


def cvt_official_model():
    # get model and replace the original fc layer with your fc layer
    model_ft = models.resnet101(pretrained=True)
    # Use this an input trace to serialize the model
    input_shape = (3, 224, 224)
    model_ft.train(False)
    orig_state_dict_keys = _unique_state_dict(model_ft).keys()
    print (orig_state_dict_keys)
    #for param in model_ft.parameters():
    #    param.requires_grad = False
    #num_ftrs = model_ft.classifier.in_features
    #model_ft.classifier = nn.Linear(num_ftrs, num_class)

    # multi-GPU
    #model_ft = torch.nn.DataParallel(model_ft).cuda()
    # dummy_input = Variable(torch.randn(1, 3, 224, 224)).cuda()
    # Export the model to an ONNX file
    dummy_input = Variable(torch.randn(1, *input_shape))
    graph, torch_out =  _trace_and_get_graph_from_model(model_ft,dummy_input,False)
    print (list(graph.inputs()))
    #print (list(graph.outputs()))
   
    #torch.onnx.export(model_ft, dummy_input, "./cvtmodels/resnet101.onnx", verbose=True)



if __name__ == '__main__':
    #cvt_my_model("./checkpoint/ckpt.t70_2018092601")
    cvt_resnext("./checkpoint/ckpt.t70_20181018")
    #cvt_my_model("./checkpoint/ckpt.t70_2018092001")
    #cvt_my_model("./checkpoint/ckpt.t70_2018092002")
    #cvt_my_model("./checkpoint/ckpt.t70_20180927")
    #cvt_official_model()