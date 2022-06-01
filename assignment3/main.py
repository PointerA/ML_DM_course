# coding=utf-8
import argparse
import networkx as nx
from fastunfolding import *
import community
from sklearn.metrics.cluster import normalized_mutual_info_score
import time

#数据和标签的输入如下：
#data_paths = ['data/karate.gml', 'data/dolphins.gml', 'data/email-Eu-core.txt']
#label_paths = ['data/karate_label.txt', 'data/dolphins_label.txt','data/email-Eu-core-department-labels.txt']

################# arg #########################
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/dolphins.gml", help="data_path is the path of the net")
parser.add_argument("--label_path", type=str, default="data/dolphins_label.txt", help="data_path is the path of the net")
parser.add_argument("--delta_Q", type=int, default=1, help="delta_Q version")
conf = parser.parse_args()

################# 读入label #########################
label_path = conf.label_path
f = open(label_path, 'r')
lines = f.readlines()
f.close()
true_label = []
for line in lines:
    n = line.split()
    if not n:
        break
    true_label.append(n[1])

################# 读数据并调用实现的算法 #########################
data_path = conf.data_path
net_G_nodes,net_G_edges = read_data(data_path)
new_G = Louvain(net_G_nodes, net_G_edges, conf.delta_Q)  #构造成一个类，传入的参数依次是点集和边集
time_start = time.time()  # 记录开始时间
partition1,Q1 = new_G.apply_method() #应用其中的方法，对社区进行分割。返回值分别是最佳的社区划分，以及对应的Q值.
time_end = time.time()  # 记录结束时间
fastunfolding_label = [0]*len(net_G_nodes)
for i,part in enumerate(partition1):
    for node in part:
        fastunfolding_label[node] = i
fastunfolding_nmi = normalized_mutual_info_score(fastunfolding_label, true_label)
print("fastunfolding_nmi:",fastunfolding_nmi)
print("fastunfolding_Q:",Q1)
print("fastunfolding time:%.3fs"  % (time_end - time_start))# 计算的时间差为程序的执行时间，单位为秒/s

################# 调用python-louvain的算法 #########################
if (data_path[-3:] == "txt"):  #如果文件是txt格式
    net_G = nx.read_edgelist(data_path) 
elif (data_path[-3:] == "gml"):  #如果文件是gml格式
    net_G  = nx.read_gml(data_path,label="id")
time_start = time.time()  # 记录开始时间
partition2 = community.best_partition(net_G)
Q2 = community.modularity(partition2,net_G)
time_end = time.time()  # 记录结束时间
pylouvain_label = []
for i in range(len(net_G_nodes)):
    if(data_path[-3:] == "txt"):
        pylouvain_label.append(partition2[str(i)])      #email-Eu-core.txt
    elif(data_path[-5:] == "e.gml"):
        pylouvain_label.append(partition2[i+1])      #karate.gml
    else:
        pylouvain_label.append(partition2[i])      #dolphins.gml
    
pylouvain_nmi = normalized_mutual_info_score(pylouvain_label, true_label)
print("pylouvain_nmi:",pylouvain_nmi)
print("pylouvain_Q:",Q2)
print("pylouvain time:%.3fs" % (time_end - time_start))# 计算的时间差为程序的执行时间，单位为秒/s
