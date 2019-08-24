# -*- coding: utf-8
"""卷积自编码网络。MNIST图像编码"""
from tensorflow.examples.tutorials.mnist import input_data
from cae import ConvolutionAutoEncoder as CAE



def mnistDataLoad():
    mnist = input_data.read_data_sets ("./MNIST_data/", one_hot=True)
    print ("Training data size:{}".format(mnist.train.num_examples))
    print ("Validating data size:{}".format(mnist.validation.num_examples))
    print ("Testing data size:{}".format(mnist.test.num_examples))
    return mnist
    
    
def main():
    pass
    
    
if __name__ == "__main__":
    main()
