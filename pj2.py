import numpy as np
from hmmlearn import hmm
import csv
import random
np.random.seed(42)

x_train = open('x_train.csv', 'r', encoding='utf-8')
x_test = open('x_test.csv', 'r', encoding='utf-8')
train_reader = csv.reader(x_train)
test_reader = csv.reader(x_test)

train_size = 1000
test_size = 3000
model_list = []
components_size = 10

for i in range(0, 10):
	temp = hmm.GaussianHMM(n_components = components_size, covariance_type="diag")
	temp.n_iter=100
	temp.tol=0.01
	model_list.append(temp)

num = 0

def data_process(li):
	ret = []
	length = int(len(li)/2)
	for i in range(0, length-1):
		ret.append(li[2*i+2]-li[2*i])
		ret.append(li[2*i+3]-li[2*i+1])
	return ret

def add_mid_point(li):
	ret = []
	length = int(len(li)/2)
	for i in range(0, length-1):
		ret.append(li[2*i])
		ret.append(li[2*i+1])
		ret.append((li[2*i]+li[2*i+2])/2)
		ret.append((li[2*i+1]+li[2*i+3])/2)
	return ret

print("train start")

for line in train_reader:
	if(num%50 == 0):
		print(num)
	if(num > train_size):
		break
	answer = int(line[2])
	length = int(line[0])
	sub_list = line[3:3+length*2]
	sub_list = list(map(lambda i: float(i), sub_list))
	if(len(sub_list)==2):
		continue
	while len(sub_list)<8 : #2 point case.
		sub_list = add_mid_point(sub_list)
	sub_list = data_process(sub_list)
	while len(sub_list) < 20:
		#print(num)
		#print(line[0])
		#print(sub_list)
		sub_list = add_mid_point(sub_list)
	X = np.array(sub_list).reshape(-1, 1)
	lengths = [2] * int(len(sub_list)/2)
	model_list[answer].fit(X, lengths)
	num = num+1
answer = 0
n = 0
print("train finished")
for line in test_reader:
	if(n%50 == 0):
		print(n)
	if(n > test_size):
		break
	length = int(line[0])
	sub_list = line[3:3+length*2]
	sub_list = list(map(lambda i: float(i), sub_list))
	if(len(sub_list)==2):
		continue
	while len(sub_list)<8:
		sub_list = add_mid_point(sub_list)
	sub_list = data_process(sub_list)
	while len(sub_list) < 20:
		sub_list = add_mid_point(sub_list)
	X = np.array(sub_list).reshape(-1, 1)
	lengths = [2] * int(len(sub_list)/2)
	Y = []
	for i in range(0, 10):
		Y.append(model_list[i].score(X))
	inference = Y.index(max(Y))
	if(inference == int(line[2])):
		answer = answer + 1
	n = n+1
x_train.close()
x_test.close()

acculate = (answer / n)*100
print(acculate)
