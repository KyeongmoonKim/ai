import csv
import random

name_list = ['AABA','AAPL','AMZN','GOOGL','IBM','KO'] #file name_list


def parser(date):
	length = len(date)
	year = int(date[0:4]) - 2006
	month = int(date[5:7]) - 1
	return (year * 12 + month) #idx in row_list
	
x_train = open('x_train.csv', 'w', encoding='utf-8')
x_test = open('x_test.csv', 'w', encoding='utf-8')
train_writer = csv.writer(x_train)
test_writer = csv.writer(x_test)
for name in name_list:
	num = 0
	fr = open(name+'_2006-01-01_to_2018-01-01.csv', 'r', encoding='utf-8')
	row_list = []
	for i in range(0, 144): #2006.01~2017.12
		row_list.append([i])
	rdr = csv.reader(fr)
	for line in rdr: #divide idx accordng to date
		if(line[0]=="-1"): break
		if(num!=0):
			idx = parser(line[0])
			row_list[idx].append([num, line[0], line[1]])
		num = num+1
	for i in range(0, 1000): #train data making
		data = []
		for j in range(0, 144):
			n = len(row_list[j])
			idx = random.randrange(1, n)
			data.append(row_list[j][idx][1])
			data.append(row_list[j][idx][2])
		data.append(name)
		train_writer.writerow(data)
	for i in range(0, 1000): #test data making
		data = []
		for j in range(0, 144):
			n = len(row_list[j])
			idx = random.randrange(1, n)
			data.append(row_list[j][idx][1])
			data.append(row_list[j][idx][2])
		data.append(name)
		test_writer.writerow(data)
	fr.close()

x_train.close()
x_test.close()