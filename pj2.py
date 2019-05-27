import numpy as np
from hmmlearn import hmm
import csv
np.random.seed(42)

def get_data(input): #first version, only up and down value
	"""ret = []
	for i in range(0, 143):
		ret.append(float(input[2*i+1])-float(input[2*i+3]))
	return ret""" # original
	ret = []
	for i in range(0, 143):
		temp1 = float(input[2*i+1])
		temp2 = float(input[2*i+3])
		if(temp1 >= temp2):
			ret.append(1)
		else:
			ret.append(0)
	print(ret)
	return ret

def get_idx(str):
	if(str=="AABA"):
		return 0
	elif(str=="AAPL"):
		return 1
	elif(str=="AMZN"):
		return 2
	elif(str=="GOOGL"):
		return 3
	elif(str=="IBM"):
		return 4
	else:
		return 5

x_train = open('x_train.csv', 'r', encoding='utf-8')
x_test = open('x_test.csv', 'r', encoding='utf-8')
train_reader = csv.reader(x_train)
test_reader = csv.reader(x_test)

model_list = []
components_size = 20
st_prob = [0.05]*components_size
tr_prob = []
for i in range(0, components_size):
	tr_prob.append([0.05]*components_size)
for i in range(0, 6):
	temp = hmm.GaussianHMM(n_components = components_size, covariance_type="diag", init_params="cm", params="cmt")
	temp.startprob_ = np.array(st_prob)
	temp.transmat_ = np.array(tr_prob)
	temp.n_iter=100
	temp.tol=0.01
	model_list.append(temp)

num = 0
for line in train_reader:
	print(num)
	if(line[0]=="-1"):
		break;
	X = np.array(get_data(line)).reshape(-1, 1)
	model_list[get_idx(line[144])].fit(X)
	#print(model_list[get_idx(line[144])].transmat_)
	num = num+1
answer = 0
n = 0

for line in test_reader:
	if(line[0]=="-1"):
		break;
	X = np.array(get_data(line)).reshape(-1, 1)
	Y = []
	for i in range(0, 6):
		Y.append(model_list[i].score(X))
	inference = Y.index(max(Y))
	if(inference == get_idx(line[144])):
		answer = answer + 1
	n = n+1
x_train.close()
x_test.close()

acculate = answer / n
print(acculate)
