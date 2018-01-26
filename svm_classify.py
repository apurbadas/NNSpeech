# Classifier using SVM

#### Libraries
import numpy as np
from numpy import genfromtxt
## My Libraries
from features import mfcc
from features import logfbank
# Third-party libraries
from sklearn import svm

def svm_baseline():

    #### Change here
    ind = 0;  # 0 for mfcc,  1 for filterbank,  2 for both
    narr = np.array([13, 26, 39]); # corresponding length of feature in a frame


    #Training and testing data
    timit_data_train = genfromtxt('timit_data_1280_train.csv', delimiter=',')
    timit_vwlname_train = genfromtxt('timit_vwlname_1280_train.csv', delimiter=',')
    timit_vwlname_train[:] = [x - 1 for x in timit_vwlname_train]
    timit_data_test = genfromtxt('timit_data_1280_test.csv', delimiter=',')
    timit_vwlname_test = genfromtxt('timit_vwlname_1280_test.csv', delimiter=',')     
    timit_vwlname_test[:] = [x - 1 for x in timit_vwlname_test]

    fs = 16000
    datalen = 1280
    i=0; j=0;
    trainfeature=np.zeros((len(timit_data_train), (datalen*100/fs - 1)*narr[ind]))
    for x in timit_data_train:
        fbank_flat = logfbank(x,fs).flatten()
        mfcc_flat = mfcc(x,fs).flatten()
        if ind == 0:
            trainfeature[i,:] = mfcc_flat
        elif ind == 1:
            trainfeature[i,:] = fbank_flat
        else:
            trainfeature[i,:] = np.concatenate((mfcc_flat, fbank_flat))
        i = i+1
        
    testfeature=np.zeros((len(timit_data_test), (datalen*100/fs - 1)*narr[ind]))
    for x in timit_data_test:
        fbank_flat = logfbank(x,fs).flatten()
        mfcc_flat = mfcc(x,fs).flatten()
        if ind == 0:
            testfeature[j,:] = mfcc_flat
        elif ind == 1:
            testfeature[j,:] = fbank_flat
        else:
            testfeature[j,:] = np.concatenate((mfcc_flat, fbank_flat))
        j = j+1

    training_data = (list(trainfeature), timit_vwlname_train)
    test_data = (list(testfeature), timit_vwlname_test)


    # train
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "Using svm_baseline classifier:"
    print "%s of %s values correct.  %s percent " % (num_correct, len(test_data[1]),
        (num_correct*100)/len(test_data[1]))

if __name__ == "__main__":
    svm_baseline()

def vectorized_result(j):
    e = np.zeros((12, 1))
    e[j] = 1.0
    return e 
