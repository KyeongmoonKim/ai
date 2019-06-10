import csv
import random
import math
import numpy as np

input = open('data_pool.csv', 'r', encoding='utf-8')
input_reader = csv.reader(input)
stroke = open('stroke2.csv', 'r', encoding='utf-8')
glyph = open('glyph2.csv', 'r', encoding='utf-8')
stroke_reader = csv.reader(stroke)
glyph_reader = csv.reader(glyph)
output = open('data_pool_rabel_unsorted.csv', 'w', encoding='utf-8')
output_writer = csv.writer(output)

sort = sorted(stroke_reader, key=lambda row: int(row[4]))
stroke_pk = []
stroke_fk = []
stroke_glyph_key = []
stroke_answer = []

for line in sort:
	stroke_pk.append(int(line[0]))
	stroke_fk.append(int(line[5]))
	stroke_glyph_key.append(int(line[4]))

#for i in range(0, 200):
#(stroke_glyph_key[i])
curr = 0
for line in glyph_reader:
	idx = int(line[0])
	while stroke_glyph_key[curr] == idx:
		stroke_answer.append(int(line[10]))
		curr = curr+1
		if(curr==len(stroke_glyph_key)):
			break

curr = 0

num = 0
for line in input_reader:
	ret = line[0:2]
	ret2 = line[2:2*int(line[0])+2]
	idx = int(line[1])
	if(idx==25913):
		continue
	stroke_idx = stroke_pk.index(idx)
	ret.append(stroke_answer[stroke_idx])
	ret.append(stroke_glyph_key[stroke_idx])
	ret.append(stroke_fk[stroke_idx])
	curr = curr+1
	ret = ret + ret2
	output_writer.writerow(ret)
	num = num+1
input.close()
output.close()
stroke.close()
glyph.close()
