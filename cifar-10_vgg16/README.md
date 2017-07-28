# Cifar-10 VGG-16 实现
参考 http://blog.csdn.net/diligent_321/article/details/53130913 实现，修改了 1.x 版本后接口不同的问题
## 工程目录
```
-cifar-10-batches-bin  //存放数据集的文件夹，可以 https://www.cs.toronto.edu/~kriz/cifar.html 下载
    -checkpoint   //存放模型训练数据的文件夹
    -forward_prop.py  //定义了前向传播的过程以及神经网络中的参数
    -input_dataset.py  //定义了数据预处理方法
    -train.py    //定义了神经网络的训练过程
    -evaluate.py    //定义了神经网络的评估过程    
```    
## 训练方法
```python
python3 train.py
```
