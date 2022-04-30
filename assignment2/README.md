# 关于程序

## 代码简介

main.py、main2.py是主程序，调用kmeans.py和data.py文件进行训练和进行画图。

data.py是获取数据，包含读取数据、分割测试集和训练集、数据标准化。

kmeans.py是K-means模型。

## 参数

main.py是一个参数较多的main函数，可以使用下列命令行参数，来训练模型并生成nmis随迭代次数变化的曲线。

| 参数名   | 参数含义                                     | 默认值     |
| -------- | -------------------------------------------- | ---------- |
| --epochs | epochs in training process                   | 500        |
| --k      | 聚类个数k的值                                | 3          |
| --data   | 数据集路径                                   | ./dataset/ |
| --save   | 存储图像的路径                               | ./         |
| --method | 三种初始化方式("random"、"distance"、"plus") | "random"   |

main2.py是为了生成这次作业要求的曲线的main函数，可以使用下列命令行参数，在特定的数据集上训练模型并生成作业要求的两个图。

| 参数名   | 参数含义                   | 默认值     |
| -------- | -------------------------- | ---------- |
| --epochs | epochs in training process | 500        |
| --data   | 数据集路径                 | ./dataset/ |
| --save   | 存储图像的路径             | ./         |

## 运行方式

到代码存放的路径下运行main.py文件

例如 python main.py --epochs 500 --k 3 --data ./dataset/iris.xlsx --save ./result/ --method random



