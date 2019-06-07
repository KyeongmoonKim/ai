import csv
import random
import math
import numpy as np

input = open('data_rabel_pool.csv', 'r', encoding='utf-8')
input_reader = csv.reader(input)

x_train = open('x_train.csv', 'w', encoding='utf-8')
x_test = open('x_test.csv', 'w', encoding='utf-8')
train_writer = csv.writer(x_train)
test_writer = csv.writer(x_test)
num = 0

for line in input_reader:
	if(num < 15000):
		train_writer.writerow(line)
	else:
		test_writer.writerow(line)
	num = num+1
input.close()
x_train.close()
x_test.close()
