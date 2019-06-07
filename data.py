import csv
import random
import math
import numpy as np

input = open('sorted_touch.csv', 'r', encoding='utf-8')
input_reader = csv.reader(input)
output = open('data_pool.csv', 'w', encoding='utf-8')
output_writer = csv.writer(output)

num = 0
length = 0
curr = 1
ret = [1]
for line in input_reader:
	if(int(line[0]) == curr):
		ret.append(line[2])
		ret.append(line[3])
		length = length + 1
		continue
	#not same.
	ret = [length] + ret
	output_writer.writerow(ret)
	curr = int(line[0])
	ret = [curr]
	ret.append(line[2])
	ret.append(line[3])
	length = 1
ret = [length] + ret
output_writer.writerow(ret)
input.close()
output.close()
