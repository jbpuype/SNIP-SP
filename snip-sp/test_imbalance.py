import os
import sys
PATH = os.getcwd()
sys.path.insert(1, 'PATH')
import mnist
import mnist_fashion
import cifar
import cifar_100

def counter(list):
    d={}
    for i in list:
        if i in d.keys():
            d[i]+=1
        else:
            d[i]=1
    return d

#Check for each class the number of elements in the training set and test set, so one can check if a dataset is balanced
def check(dataset):
    label_train = dataset['train']['label']
    label_test = dataset['test']['label']
    print(counter(label_train), counter(label_test))

dataset = mnist.read_data("./MNIST")
check(dataset)

dataset = mnist_fashion.read_data("./MNIST-fashion")
check(dataset)

dataset = cifar.read_data("./cifar-10")
check(dataset)

dataset = cifar_100.read_data("./cifar-100")
check(dataset)
