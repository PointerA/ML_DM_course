import numpy as np
from tqdm import tqdm
from model import *
from data import *
import matplotlib.pyplot as plt
import argparse as arg

################# arg #########################
parser = arg.ArgumentParser()

parser.add_argument("--epochs", type=int, default=2500, help="epochs in training process")
parser.add_argument("--lr", type=float, default=0.3, help="learning rate")
parser.add_argument("--lamda", type=float, default=0.0, help="the hyper-parameter used to control the influence")
parser.add_argument("--save", type=str, default="./", help="figure save path")

conf = parser.parse_args()


################# load data #####################
train_x, test_x, train_y, test_y, diction = readData(file='Dry_Bean_Dataset.xlsx', seed=5, proportion=0.7)

'''
#[X]->[X,X^2,...,X^(max_pow)]基函数
max_pow = 3
for i in range(2,max_pow+1):
    train_x = np.hstack((train_x,np.power(train_x,i)))
    test_x = np.hstack((test_x,np.power(test_x,i)))
''' 

'''
#[X]->[sin(X)]基函数
train_x = np.sin(train_x)
test_x = np.sin(test_x)
'''

b = np.ones(train_x.shape[0])
train_x = np.c_[train_x, b]
b = np.ones(test_x.shape[0])
test_x = np.c_[test_x, b]

classes = len(diction)
dim = train_x.shape[1]
onehot_y = to_onehot(train_y, classes)
test_onehot = to_onehot(test_y, classes)

################# train #####################
np.random.seed(5)
net = LinearClassfier(dim, classes, conf.lamda)
train_accs, train_losses, test_accs, test_losses = [], [], [], []
best_acc = 0
best_w = np.zeros((dim, classes))

with tqdm(total=conf.epochs) as pbar:
    grad_sum = np.zeros((dim, classes))
    for epoch in range(conf.epochs):
        y_hat = net(train_x)
        loss, acc, grad = crossEntropyLoss(train_x, y_hat, onehot_y)
        grad_sum += np.square(grad)
        net.update(conf.lr, grad, grad_sum)
        test_y_hat = net(test_x)
        test_loss, test_acc, _ = crossEntropyLoss(test_x, test_y_hat, test_onehot)
        
        train_accs.append(acc), test_accs.append(test_acc)
        train_losses.append(loss), test_losses.append(test_loss)
        if test_acc > best_acc:
            best_acc = test_acc
            best_w = net.getweight()
        pbar.set_description(f"train loss: {loss:.4f} | train acc: {acc:.4f} | test acc: {test_acc:.4f}")
        pbar.update()

####################### print plot ##########################
plt.plot(range(conf.epochs), train_losses, label = "train")
plt.plot(range(conf.epochs), test_losses, label = "test")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig(conf.save+"losses.png", format='png') 
plt.close()

plt.plot(range(conf.epochs), train_accs, label = "train")
plt.plot(range(conf.epochs), test_accs, label = "test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig(conf.save+"accs.png", format='png')
plt.close()

print('best acc: ', best_acc)