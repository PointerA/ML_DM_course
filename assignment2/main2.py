from cProfile import label
import numpy as np
from kmeans import kmeans
from data import *
import argparse as arg
import matplotlib.pyplot as plt

################# arg #########################
parser = arg.ArgumentParser()
parser.add_argument("--epochs", type=int, default=500, help="epochs in training process")
parser.add_argument("--data", type=str, default="./dataset/iris.xlsx", help="data path")
parser.add_argument("--save", type=str, default="./pic/iris_", help="figure save path")
conf = parser.parse_args()

################# Fig1 #########################
data_x,data_y,Klength = readData(conf.data)

model = kmeans(Klength, "random")
J1, nmis1 = model.fit(data_x, data_y, conf.epochs)
model = kmeans(Klength, "distance")
J2, nmis2 = model.fit(data_x, data_y, conf.epochs)
model = kmeans(Klength, "plus")
J3, nmis3 = model.fit(data_x, data_y, conf.epochs)

plt.plot(range(len(nmis1)), nmis1, label="random")
plt.plot(range(len(nmis2)), nmis2, label="distance")
plt.plot(range(len(nmis3)), nmis3, label="plus")
plt.xlabel("epoch")
plt.ylabel("nmi")
plt.legend()
plt.savefig(conf.save+"nmis.png", format='png') 
plt.close()

################# Fig2 #########################
J1s, J2s, J3s = [],[],[]
shangxian = int(50)
for i in range(2, shangxian):
    model = kmeans(i, "random")
    J1, _ = model.fit(data_x, data_y, conf.epochs)
    J1s.append(J1[len(J1)-1])
    model = kmeans(i, "distance")
    J2, _ = model.fit(data_x, data_y, conf.epochs)
    J2s.append(J2[len(J2)-1])
    model = kmeans(i, "plus")
    J3, _ = model.fit(data_x, data_y, conf.epochs)
    J3s.append(J3[len(J3)-1])

plt.plot(range(2, shangxian), J1s, label="random")
plt.plot(range(2, shangxian), J2s, label="distance")
plt.plot(range(2, shangxian), J3s, label="plus")
plt.xlabel("K")
plt.ylabel("J")
plt.legend()
plt.savefig(conf.save+"Js.png", format='png') 
plt.close()