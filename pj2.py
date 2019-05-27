import numpy as np
from hmmlearn import hmm
import csv

x_train = open('x_train.csv', 'r', encoding='utf-8')
x_test = open('x_test.csv', 'r', encoding='utf-8')
train_reader = csv.reader(x_train)
test_reader = csv.reader(x_test)

model_list = []
for i in range(0, 6):
	model_list.append(hmm.GaussianHMM(n_components = 20, covariance_type="full"))

x_train.close()
x_test.close()
