import csv
import random
import math
import numpy as np

input = open('sorted_touch.csv', 'r', encoding='utf-8')
input_reader = csv.reader(input)
output = open('data_pool.csv', 'w', encoding='utf-8')
output_writer = csv.writer(output)

num = 0
curr = 1
ret = [1]
for line in input_reader:
	if(int(line[0]) == curr):
		ret.append(line[2])
		ret.append(line[3])
		continue
	#not same.
	output_writer.writerow(ret)
	curr = curr+1
	ret = [curr]
	ret.append(line[2])
	ret.append(line[3])

input.close()
output.close()
