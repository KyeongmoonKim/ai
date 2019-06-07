import csv
import random
import math
import numpy as np

input = open('data_pool_rabel_sorted.csv', 'r', encoding='utf-8')
input_reader = csv.reader(input)
output = open('data_pool_rabel.csv', 'w', encoding='utf-8')
output_writer = csv.writer(output)

check = 0
num = 0
length = 0
curr = -1
ret = []
for line in input_reader: #0 : length, #1 : discared, #2: rabel, #3 : id #4 : in id order, after is zum/
	if(check == 0): #real first
		check = 1
		length = int(line[0])
		ret = [int(line[3]), int(line[2])]
		curr = int(line[3])
		temp = line[5:5+int(line[0])*2]
		ret = ret + temp
		continue
	if(int(line[3]) == curr):
		length = length + int(line[0])
		temp = line[5:5+int(line[0])*2]
		ret = ret + temp
		continue
	#not same.
	ret = [length] + ret
	output_writer.writerow(ret)
	curr = int(line[3])
	ret = [int(line[3]), int(line[2])]
	temp = line[5:5+int(line[0])*2]
	ret = ret + temp
	length = int(line[0])
ret = [length] + ret
output_writer.writerow(ret)
input.close()
output.close()
