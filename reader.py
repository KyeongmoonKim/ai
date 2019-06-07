import csv
import random
import math
import numpy as np

x_train = open('data_pool_rabel.csv', 'r', encoding='utf-8')
train_reader = csv.reader(x_train)
num = 0

for line in train_reader:
	if(int(line[1]) == 5):
		print(line)
		break;
	num = num+1
x_train.close()
