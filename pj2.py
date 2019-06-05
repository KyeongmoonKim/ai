import numpy as np
from hmmlearn import hmm
import csv
import random
np.random.seed(42)

x_train = open('x_train.csv', 'r', encoding='utf-8')
x_test = open('mnist_test.csv', 'r', encoding='utf-8')
train_reader = csv.reader(x_train)
test_reader = csv.reader(x_test)

train_size = 1000
test_size = 1000
model_list = []
components_size = 20
my_filter1 = [0, 0, 1.0, 1.0] #horizon
my_filter2 = [0, 1.0, 0, 1.0] #vertical
my_filter_size = 2;
my_target_size = 28;

def conv_one_step(target, filter, row, col, target_size, filter_size): #(row, col) of target, square t_s*t_s, f_s*f_s
	ret = 0
	for i in range(0, filter_size):
		for j in range(0, filter_size):
			ret = ret + target[(row + i) * target_size + col+j] * filter[i * filter_size+j]
	return ret
			
def conv(target, filter, target_size, filter_size):
	ret = []
	n = target_size - filter_size + 1
	for i in range(0, n):
		for j in range(0, n):
			value = conv_one_step(target, filter, i, j, target_size, filter_size)
			ret.append([value])
	return ret

for i in range(0, 10):
	temp = hmm.GaussianHMM(n_components = components_size, covariance_type="diag")
	temp.n_iter=100
	temp.tol=0.01
	model_list.append(temp)

num = 0

def pair_list(list1, list2): #list1, list2 same len
	if(len(list1)!=len(list2)):
		return []
	ret = []
	for i in range(0, len(list1)):
		ret.append(list1[i])
		ret.append(list2[i])
	return ret

lengths = [2]*676


print("train start")

for line in train_reader:
	if(num==0):
		num = num+1
		continue
	if(num > train_size):
		break
	if(num%100==0):
		print(num)
	sub_list = line[1:len(line)]
	sub_list = list(map(lambda i : int(i), sub_list))
	X1 = conv(sub_list, my_filter1, my_target_size, my_filter_size)
	X2 = conv(sub_list, my_filter2, my_target_size, my_filter_size)
	X = np.array(pair_list(X1,X2))
	model_list[int(line[0])].fit(X, lengths)
	num = num+1
answer = 0
n = 0
print("train finished")
for line in test_reader:
	if(n==0):
		n = n+1
		continue
	if(n > test_size):
		break
	if(n%100==0):
		print(n)
	sub_list = line[1:len(line)]
	sub_list = list(map(lambda i : int(i), sub_list))
	X1 = conv(sub_list, my_filter1, my_target_size, my_filter_size)
	X2 = conv(sub_list, my_filter2, my_target_size, my_filter_size)
	X = np.array(pair_list(X1,X2))
	Y = []
	for i in range(0, 10):
		Y.append(model_list[i].score(X, lengths))
	inference = Y.index(max(Y))
	if(inference == int(line[0])):
		answer = answer + 1
	n = n+1
x_train.close()
x_test.close()

acculate = (answer / n)*100
print(acculate)
