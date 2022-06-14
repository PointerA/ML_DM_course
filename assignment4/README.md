# 关于程序

User-based CF和Item-based CF两个版本的协同滤波算法

## 代码简介

为了方便同时跑出多个结果，因此写了4个python文件，它们之间有许多部分是重复的。

共有4个python文件：

| 文件名   | 版本          | 数据集                         |
| -------- | ------------- | ------------------------------ |
| main1.py | User-based CF | MovieLens Latest Small Dataset |
| main2.py | User-based CF | MovieLens 100K Dataset         |
| main3.py | Item-based CF | MovieLens Latest Small Dataset |
| main4.py | Item-based CF | MovieLens 100K Dataset         |

2个数据集：

- ratings.csv：MovieLens Latest Small Dataset
- ratings2.csv：MovieLens 100K Dataset

## 运行方式

到代码存放的路径下运行main.py文件

例如 python main1.py



由于用矩阵存储完整的数据集，因此需要内存不少于4g。



