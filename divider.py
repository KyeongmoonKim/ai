import csv
import random
import math
import numpy as np

input = open('data_pool.csv', 'r', encoding='utf-8')
input_reader = csv.reader(data_pool.csv)

x_train = open('x
temp_train = open('x_train.csv', 'w', encoding='utf-8')
temp_writer = csv.writer(temp_train)
num = 0

for line in input_reader:
	if(num == 5000):
		break
	temp_writer.writerow(line)
	num = num+1
x_train.close()
