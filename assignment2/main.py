import numpy as np
from kmeans import kmeans
from data import *
import argparse as arg
import matplotlib.pyplot as plt

################# arg #########################
parser = arg.ArgumentParser()

parser.add_argument("--epochs", type=int, default=500, help="epochs in training process")
parser.add_argument("--k", type=float, default=3, help="k value")
parser.add_argument("--method", type=str, default="random", help="init method")
parser.add_argument("--data", type=str, default="./dataset/iris.xlsx", help="data path")
parser.add_argument("--save", type=str, default="./", help="figure save path")

conf = parser.parse_args()
data_x,data_y,_ = readData(conf.data)

model = kmeans(conf.k, conf.method)

sum_dists, nmis = model.fit(data_x, data_y, conf.epochs)

plt.plot(range(len(nmis)), nmis)
plt.xlabel("epoch")
plt.ylabel("nmi")
plt.savefig(conf.save+"nmis.png", format='png') 
plt.close()
