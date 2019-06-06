import csv
import operator
import random
import math
import numpy as np

unsorted_touch = open('unsorted_touch.csv', 'r', encoding='utf-8')
unsorted_touch_reader = csv.reader(unsorted_touch)
sorted_touch = open('sorted_touch.csv', 'w', encoding='utf-8')
sorted_touch_writer = csv.writer(sorted_touch)

num = 0
temp = 0
sort = sorted(unsorted_touch_reader, key=lambda row: (int(row[4]), float(row[6])))

temp_max = 0

for line in sort:
	#sorted_touch_writer.writerow(line)
	if(int(line[4]) == 2):
		temp = temp+1
	temp_max = max(temp_max, int(line[4]))
	ret = [line[4], line[6], line[7], line[8]]
	sorted_touch_writer.writerow(ret)
	num = num+1

print(temp_max)
print(temp)
