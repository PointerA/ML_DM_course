# 关于程序

## 代码简介

main.py是主程序，调用另外两个文件进行训练和进行画图。

data.py是获取数据，包含读取数据、分割测试集和训练集、数据标准化。

model.py是线性分类器模型，softmax函数、交叉熵、梯度下降等等都被包含在此文件内。

## 参数

| 参数名   | 参数含义                   | 默认值 |
| -------- | -------------------------- | ------ |
| --epochs | epochs in training process | 2500   |
| --lr     | learning rate              | 0.3    |
| --lamda  | 正则化系数                 | 0.0    |
| --save   | 存储图像的路径             | ./     |

## 运行方式

到代码存放的路径下运行main.py文件

例如 python main.py --epochs 2500 --lr 0.3 --lamda 0.0 --save ./result/



## 一些可能需要在代码内修改的地方

model.py文件内 class LinearClassfier 下的update方法，19、20行分别是包含L1范数和L2范数正则化项的参数更新代码，若想使用正则化项，可将注释去掉。

为保证结果的可复现，main.py函数的第20行和第47行可以设置随机数种子。