import csv
import random
import math
import numpy as np

touch = open('touch.csv', 'r', encoding='utf-8')
glyph = open('glyph.csv', 'r', encoding='utf-8')
stroke = open('stroke.csv', 'r', encoding='utf-8')
touch_reader = csv.reader(touch)
glyph_reader = csv.reader(glyph)
stroke_reader = csv.reader(stroke)
temp_train = open('unsorted_touch.csv', 'w', encoding='utf-8')
stroke2 = open('stroke2.csv', 'w', encoding='utf-8')
glyph2 = open('glyph2.csv', 'w', encoding='utf-8')
touch_writer = csv.writer(temp_train)
stroke2_writer = csv.writer(stroke2)
glyph2_writer = csv.writer(glyph2)

num = 0

for line in touch_reader:
	if(num!=0):
		touch_writer.writerow(line)
	num = num+1
num = 0

for line in stroke_reader:
	if(num!=0):
		stroke2_writer.writerow(line)
	num = num+1

num = 0

for line in glyph_reader:
	if(num!=0):
		glyph2_writer.writerow(line)
	num = num+1

touch.close()
temp_train.close()
stroke.close()
stroke2.close()
glyph.close()
glyph2.close()
