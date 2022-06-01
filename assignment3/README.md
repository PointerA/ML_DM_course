# 关于程序

社区网络发现

## 代码简介

main.py、是主程序，调用fastunfolding.py文件进行社区划分。

fastunfolding.py实现了Louvain算法。

## 参数

main.py是一个含参数的main函数，可以使用下列命令行参数。

| 参数名       | 参数含义           | 默认值                    |
| ------------ | ------------------ | ------------------------- |
| --data_path  | 数据集路径         | "data/dolphins.gml"       |
| --label_path | 标签的路径         | "data/dolphins_label.txt" |
| --delta_Q    | 两种计算方式(1or2) | 1                         |

```python
#数据和标签的可以输入如下：
#data_paths = ['data/karate.gml', 'data/dolphins.gml', 'data/email-Eu-core.txt']
#label_paths = ['data/karate_label.txt', 'data/dolphins_label.txt','data/email-Eu-core-department-labels.txt']
```

## 运行方式

到代码存放的路径下运行main.py文件

例如 python main.py --data_path data/dolphins.gml --label_path data/dolphins_label.txt --delta_Q 1



