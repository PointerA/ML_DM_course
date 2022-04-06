import numpy as np


class LinearClassfier:
    def __init__(self, dim, classes, lamda=0.0):
        self.dim = dim
        self.classes = classes
        self.W = np.random.randn(dim, classes)
        self.lamda = lamda

    def __call__(self, x):
        # X: [N, dim] W: [dim, Classes]
        # mul(X, W): [N, Classes]
        return softmax(np.matmul(x, self.W))

    def update(self, lr, grad, grad_sum):
        self.W -= lr / np.sqrt(grad_sum + 1e-16) * grad
        
        #self.W = self.W - lr * grad - self.lamda * lr * np.sign(self.W)   #1-norm
        #self.W = (1 - self.lamda * lr) * self.W - lr * grad    #2-norm
        
    def getweight(self):
        return self.W


# calculate the cross entropy loss
def crossEntropyLoss(x, y_hat, onehot_y):
    loss = -(np.log(y_hat + 1e-16) * onehot_y).sum() / y_hat.shape[0]
    acc = cal_acc(y_hat, onehot_y)
    grad = np.dot(x.T, (y_hat - onehot_y)) / y_hat.shape[0]
    return loss, acc, grad


# the softmax function
def softmax(z):
    z2 = np.max(z, axis=1)
    z2 = z2.reshape(-1, 1)
    z -= z2   # To advoid the value is too big
    z = np.exp(z)
    z3 = np.sum(z, axis=1, keepdims=True)
    return z / z3


#change the labels to on hot vector
def to_onehot(y_label, classes):
    onehot_label = np.eye(classes)[y_label]
    return onehot_label


#calculate the acc
def cal_acc(y_hat, onehot_y):
    y_predict = np.argmax(y_hat, axis = -1)
    y_label = np.argmax(onehot_y, axis = -1)
    return (y_predict == y_label).sum() / onehot_y.shape[0]