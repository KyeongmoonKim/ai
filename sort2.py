import csv
import operator
import random
import math
import numpy as np

unsorted_touch = open('data_pool_rabel_unsorted.csv', 'r', encoding='utf-8')
unsorted_touch_reader = csv.reader(unsorted_touch)
sorted_touch = open('data_pool_rabel_sorted.csv', 'w', encoding='utf-8')
sorted_touch_writer = csv.writer(sorted_touch)

num = 0
temp = 0
sort = sorted(unsorted_touch_reader, key=lambda row: (int(row[3]), float(row[4])))

temp_max = 0

for line in sort:
	sorted_touch_writer.writerow(line)
	num = num+1
