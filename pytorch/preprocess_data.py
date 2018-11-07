#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,shutil
import sys
import random
import threading

synset_labels="/mnt/xh_data/datasets/digestive8/train.txt"
dataset_dir="/mnt/xh_data/datasets/digestive8/train"
train_dir="/home/gfkd/bito8/train_dir"
test_dir="/home/gfkd/bito8/test_dir"
validation_dir="/home/gfkd/bito8/validation_dir"

train_percent=0.8
test_percent=0.1
validation_percent=0.1

def mkdir(path):     #判断是否存在指定文件夹，不存在则创建
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print (path)
        print  (' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path)
        print  (' 目录已存在')
        return False

def moveFile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!" % (srcfile))
    else:
        # fpath,fname=os.path.split(dstfile)
        shutil.move(srcfile,dstfile)
        print ("move %s -> %s" % (srcfile,dstfile))

def copyFile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!" % (srcfile))
    else:
        # fpath,fname=os.path.split(dstfile)
        shutil.copyfile(srcfile,dstfile)
        print ("copy %s -> %s" % (srcfile,dstfile))

def splitLabel(line):
    # 
    arr=line.split(" ")
    filename=arr[0].strip()
    label=arr[1].strip()
    return filename,label

def builder(lst,dst_dir):
    label=os.path.join(dst_dir,"labels")
    with open(label,'w') as fp:
        for item in lst:
            fp.write(item.strip()+"\n")
            filename,label=splitLabel(item.strip())
            srcfile=os.path.join(dataset_dir,filename)
            dstfile=os.path.join(dst_dir,filename)
            copyFile(srcfile,dstfile)

def process():
    print ("processing %s" % (synset_labels))
    mkdir(train_dir)
    mkdir(test_dir)
    mkdir(validation_dir)
    with open(synset_labels) as f:
        lines = f.readlines()
        size = len(lines)
        #乱序
        random.shuffle(lines)
        #训练集大小
        train_size=int(train_percent*size)
        train_list=lines[0:train_size]

        #测试集大小
        test_size=int(test_percent*size)
        test_list=lines[train_size:train_size+test_size]

        #验证集大小
        validation_list=lines[train_size+test_size:size]

        print (size)
        print ("train_list %d:%d" % (0,train_size))
        print ("test_list %d:%d" % (train_size,train_size+test_size))
        print ("validation_list %d:%d" % (train_size+test_size,size))

        print (len(train_list))
        print (len(test_list))
        print (len(validation_list))

        # t1=threading.Thread(target=builder,name='builder',args=(train_list,train_dir))
        # t2=threading.Thread(target=builder,name='builder',args=(test_list,test_dir))
        # t3=threading.Thread(target=builder,name='builder',args=(validation_list,validation_dir))
        # t1.start()
        # t2.start()
        # t3.start()
        # t1.join()
        # t2.join()
        # t3.join()

        builder(train_list,train_dir)
        builder(test_list,test_dir)
        builder(validation_list,validation_dir)

    print ("process completed.")

def main():
    process()

if __name__ == '__main__':
  	main()