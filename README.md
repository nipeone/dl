# dl
deep learning of pytorch 、mxnet etc.
> 针对同一个数据集，分别用pytorch、mxnet等方式进行训练、验证及预测，设计到pytorch转mxnet模型的处理方式
## 提升精度的方式
1. data_augment 包括图片的随机裁剪、随机反转及随机旋转角度
2. mixup方式处理数据集
3. tta
4. 多模型融合，将皮尔斯系数相关性最大的模型进行融合
5. 训练集增强，将原始图片随机成多张图片，然后方差值最大的图片