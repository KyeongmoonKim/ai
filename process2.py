import csv
import random
import math
import numpy as np

x_train = open('touch.csv', 'r', encoding='utf-8')
train_reader = csv.reader(x_train)
temp_train = open('unsorted_touch.csv', 'w', encoding='utf-8')
temp_writer = csv.writer(temp_train)

num = 0

for line in train_reader:
	if(num!=0):
		temp_writer.writerow(line)
	num = num+1
x_train.close()
