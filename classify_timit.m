clear all;close all;clc;

train_data = csvread('timit_data_1280_train.csv');
test_data = csvread('timit_data_1280_test.csv');
train_vwlname = csvread('timit_vwlname_1280_train.csv');
test_vwlname = csvread('timit_vwlname_1280_test.csv');

% result = classify(test_data,train_data,train_vwlname);
idx = knnsearch(train_data,test_data,'k',7);
result_knn = mode(train_vwlname(idx),2);
 
% accr_percent = (length(test_vwlname)-nnz(test_vwlname-result))/length(test_vwlname)*100
accr_percent_knn = (length(test_vwlname)-nnz(test_vwlname-result_knn))/length(test_vwlname)*100