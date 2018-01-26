# Main file for classification using NN

import theano_feature
from theano_feature import Network
from theano_feature import FullyConnectedLayer, SoftmaxLayer
import numpy as np


#### Change here
ind = 2;  # 0 for mfcc,  1 for filterbank,  2 for both
epochs = 500
mini_batch_size = 10
eta = 0.002
narr = np.array([13, 26, 39]);


training_data, validation_data, test_data = theano_feature.load_data_shared(ind)

fs = 16000
datalen = 1280;
frame = datalen*100/fs - 1;
net = Network([
        FullyConnectedLayer(n_in=narr[ind]*frame, n_out=100),
        SoftmaxLayer(n_in=100, n_out=12)], mini_batch_size)

net.SGD(training_data, epochs, mini_batch_size, eta, 
            validation_data, test_data)
