import numpy as np
from hmmlearn import hmm
import csv
import datetime
np.random.seed(42)

x_train = open('x_train.csv', 'r', encoding='utf-8')
x_test = open('mnist_test.csv', 'r', encoding='utf-8')
train_reader = csv.reader(x_train)
test_reader = csv.reader(x_test)

train_size = 1000
test_size = 1000
model_list = []
components_size = 20

for i in range(0, 10):
	temp = hmm.GaussianHMM(n_components = components_size, covariance_type="diag")
	temp.n_iter=100
	temp.tol=0.01
	model_list.append(temp)

num = 0
#lengths = [2]*143

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
	X = np.array(sub_list).reshape(-1, 1)
	#print(X)
	model_list[int(line[0])].fit(X)
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
	X = np.array(sub_list).reshape(-1, 1)
	Y = []
	for i in range(0, 10):
		Y.append(model_list[i].score(X))
	inference = Y.index(max(Y))
	if(inference == int(line[0])):
		answer = answer + 1
	n = n+1
x_train.close()
x_test.close()

acculate = (answer / n)*100
print(acculate)
