import numpy as np
from hmmlearn import hmm
import csv
import random
np.random.seed(42)
import math
import copy

x_train = open('x_train.csv', 'r', encoding='utf-8')
x_test = open('x_test.csv', 'r', encoding='utf-8')
train_reader = csv.reader(x_train)
test_reader = csv.reader(x_test)

train_size = 1000
test_size = 3000
model_list = []
components_size = 10

for i in range(0, 10): # 2*i : original order, 2*i+1: reversed order
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
	ret.append(li[len(li)-2])
	ret.append(li[len(li)-1])
	return ret

def add_mid_vector_with_noise(li):
	ret = []
	length = int(len(li)/2)
	for i in range(0, length-1):
		ret.append(li[2*i])
		ret.append(li[2*i+1])
		temp1 = (li[2*i]+li[2*i+2])/2
		temp2 = (li[2*i+1]+li[2*i+3])/2
		v_x = 0.9998*temp1-0.0174*temp2
		v_y = 0.0174*temp1+0.9998*temp2
		ret.append(v_x)
		ret.append(v_y)
	ret.append(li[len(li)-2])
	ret.append(li[len(li)-1])
	return ret

def zero_delete(li): #zero vector delete
	ret = []
	length = int(len(li)/2)
	for i in range(0, length):
		if(li[2*i] == 0.0 and li[2*i+1] == 0.0):
			continue
		else:
			ret.append(li[2*i])
			ret.append(li[2*i+1])
	return ret

def normalize(li): #normalizing, and zero vector delte
	ret = []
	length = int(len(li)/2)
	for i in range(0, length):
		temp = math.sqrt(li[2*i]*li[2*i] + li[2*i+1]*li[2*i+1])
		if(temp==0.0):
			continue
		else:
			ret.append(li[2*i]/temp)
			ret.append(li[2*i+1]/temp)
	return ret

print("train start")

def data_process2(li): #axis-transmation use, angle calculation, if (zero, zero):continue
	ret = []
	length = int(len(li)/2)
	for i in range(0, length-1):
		temp1 = li[2*i]*li[2*i+2] + li[2*i+1]*li[2*i+3]
		temp2 = li[2*i]*li[2*i+3] - li[2*i+1]*li[2*i+2]
		ret.append(temp1)
		ret.append(temp2)
		#ret.append(math.atan2(temp2, temp1))
	return ret

def reverse(li): #reverse point list, we train the machine with using point array, and reversed point array
	ret = copy.deepcopy(li)
	ret.reverse()
	length = int(len(li)/2)
	for i in range(0, length):
		temp = ret[2*i+1]
		ret[2*i+1] = ret[2*i]
		ret[2*i] = temp
	return ret

not_passed = 0
for line in train_reader: #train part
	if(num%50 == 0):
		print(num)
	if(num > train_size):
		break
	answer = int(line[2])
	length = int(line[0])
	sub_list = line[3:3+length*2]
	sub_list = list(map(lambda i: float(i), sub_list))
	if(len(sub_list)==2): #only one point : ignore
		not_passed = not_passed+1
		continue
	while len(sub_list)<8 : #2 point case.
		sub_list = add_mid_point(sub_list)
	sub_list_rev = reverse(sub_list)
	sub_list = data_process(sub_list) #vectorize
	sub_list_rev = data_process(sub_list_rev)
	sub_list = zero_delete(sub_list) #zero-vector delete
	sub_list_rev = zero_delete(sub_list_rev)
#	print(sub_list)
#	print(sub_list_rev)
	if(len(sub_list)<2): #no vetor
		not_passed = not_passed+1
		continue
	if(len(sub_list)==2): #only one vector, because in vector duplicate 1 is ok.
		sub_list.append(sub_list[0])
		sub_list.append(sub_list[1])
		sub_list_rev.append(sub_list_rev[0])
		sub_list_rev.append(sub_list_rev[1])
	while len(sub_list) < 30:
#		sub_list = add_mid_vector_with_noise(sub_list)
#		sub_list_rev = add_mid_vector_with_noise(sub_list_rev)
		sub_list = add_mid_point(sub_list)
		sub_list_rev = add_mid_point(sub_list_rev)
	sub_list = normalize(sub_list)
	sub_list_rev = normalize(sub_list_rev)
	sub_list = data_process2(sub_list)
	sub_list_rev = data_process2(sub_list_rev)
	X = np.array(sub_list).reshape(-1, 1)
	lengths = [2] * int(len(sub_list)/2)
	model_list[answer].fit(X,lengths) #lengths
	X = np.array(sub_list_rev).reshape(-1,1)
	model_list[answer].fit(X,lengths)
	num = num+1

answer = 0
n = 0
print("train finished")
print("not passed is ")
print(not_passed)


for line in test_reader: #test part
	if(n%50 == 0):
		print(n)
	if(n > test_size):
		break
	length = int(line[0])
	sub_list = line[3:3+length*2]
	sub_list = list(map(lambda i: float(i), sub_list))
	if(len(sub_list)==2): #only one point : ignore
		not_passed = not_passed+1
		continue
	while len(sub_list)<8 : #2 point case.
		sub_list = add_mid_point(sub_list)
	sub_list = data_process(sub_list) #vectorize
	sub_list = zero_delete(sub_list) #zero-vector delete
	if(len(sub_list)<2): #no vetor
		not_passed = not_passed+1
		continue
	if(len(sub_list)==2): #only one vector, because in vector duplicate 1 is ok.
		sub_list.append(sub_list[0])
		sub_list.append(sub_list[1])
	while len(sub_list) < 30:
#		sub_list = add_mid_vector_with_noise(sub_list)
		sub_list = add_mid_point(sub_list)
	sub_list = normalize(sub_list)
	sub_list = data_process2(sub_list)
	X = np.array(sub_list).reshape(-1, 1)
	lengths = [2] * int(len(sub_list)/2)
	Y = []
	for i in range(0, 10):
		Y.append(model_list[i].score(X, lengths)) #lengths
	inference = Y.index(max(Y))
	#inference = int(inference/2)
	if(inference == int(line[2])):
		answer = answer + 1
	n = n+1
x_train.close()
x_test.close()

acculate = (answer / n)*100
print(acculate)
