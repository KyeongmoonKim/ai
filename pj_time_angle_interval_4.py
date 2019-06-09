import numpy as np
from hmmlearn import hmm
import csv
import random
np.random.seed(42)
import math
import matplotlib.pyplot as plt

#point mid insertion and cut

x_train = open('x_train.csv', 'r', encoding='utf-8')
result = open('result.csv', 'w', encoding='utf-8')
train_reader = csv.reader(x_train)
result_writer = csv.writer(result)

train_size = 14000
test_size = 1000
model_list = []
components_size = 6

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_ylim([0,100])
ax.set_xlim([0, 15000])

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

def add_mid_vector(li):
	ret = []
	length = int(len(li)/2)
	for i in range(0, length):
		temp1 = (li[2*i]*0.9998-0.0174*li[2*i+1])/2
		temp2 = (0.0174*li[2*i]+0.9998*li[2*i+1])/2
		ret.append(temp1)
		ret.append(temp2)
		ret.append(li[2*i]-temp1)
		ret.append(li[2*i+1]-temp2)
	return ret

def add_mid_vector_uniform(li):
	ret = []
	length = int(len(li)/2)
	for i in range(0, length):
		temp1 = li[2*i]/2
		temp2 = li[2*i+1]/2
		ret.append(temp1)
		ret.append(temp2)
		ret.append(li[2*i]-temp1)
		ret.append(li[2*i+1]-temp2)
	return ret

def add_mid_vector_random(li):
	ret = []
	length = int(len(li)/2)
	for i in range(0, length):
		check = random.randint(0,10)
		if(check<5):
			temp1 = (li[2*i]*0.9998-0.0174*li[2*i+1])/2
			temp2 = (0.0174*li[2*i]+0.9998*li[2*i+1])/2
			ret.append(temp1)
			ret.append(temp2)
			ret.append(li[2*i]-temp1)
			ret.append(li[2*i+1]-temp2)
		else:
			temp1 = (li[2*i]*0.9998+0.0174*li[2*i+1])/2
			temp2 = (0.9998*li[2*i+1]-0.0174*li[2*i])/2
			ret.append(temp1)
			ret.append(temp2)
			ret.append(li[2*i]-temp1)
			ret.append(li[2*i+1]-temp2)
	return ret

def random_shake(li):
	ret = []
	length = int(len(li)/2)
	for i in range(0, length):
		check = random.randint(0, 10)
		check2 = random.randint(0, 6)
		check2 = check2 * math.pi / 180.0
		x = math.cos(check2)
		y = math.sin(check2)
		if(check<5):
			temp1 = li[2*i]*x-y*li[2*i+1]
			temp2 = y*li[2*i]+x*li[2*i+1]
			ret.append(temp1)
			ret.append(temp2)
		else:
			temp1 = li[2*i]*x+y*li[2*i+1]
			temp2 = x*li[2*i+1]-y*li[2*i]
			ret.append(temp1)
			ret.append(temp2)
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

def vector_duplicate(li):
	ret = []
	if(len(li)!=2):
		print("unexpected length : the number of vectors isn't 1(in vector_duplicate)")
		return li
	ret.append(li[0])
	ret.append(li[1])
	check = random.randint(0, 2)
	if(check==0):
		ret.append(0.9986*li[0]-0.052*li[1])
		ret.append(0.052*li[0]+0.9986*li[1])
	else:
		ret.append(0.9986*li[0]+0.052*li[1])
		ret.append(0.9986*li[1]-0.052*li[0])
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

def same_point_delete(li):
	ret = []
	length = int(len(li)/2)
	for i in range(0, length-1):
		if(li[2*i]==li[2*i+2] and li[2*i+1]==li[2*i+3]):
			continue
		else:
			ret.append(li[2*i])
			ret.append(li[2*i+1])
	ret.append(li[len(li)-2])
	ret.append(li[len(li)-1])	
	return ret

def vectorize_term(li, expected_length):
	ret = []
	term = (int(len(li)/2)-1)/expected_length
	curr = 0
	next = 0
	for i in range(1, expected_length+1):
		next = curr + term
		temp1 = li[int(next)*2]-li[int(curr)*2]
		temp2 = li[int(next)*2+1]-li[int(curr)*2+1]
		ret.append(temp1)
		ret.append(temp2)
		curr = next	
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

x = [0]
y = [0]
result = []
for i in range(0, 10):
	result.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
not_passed = 0
num = 0
for line in train_reader:
	if(num%50 == 0):
		print(num)
	if(num%1000 == 0 and num > 0):
		print("not_passed num")
		print(not_passed)
		print("test start")
		x_test = open('x_test.csv', 'r', encoding='utf-8')
		test_reader = csv.reader(x_test)
		test_answer = 0
		n = 0
		for test_line in test_reader:
			if((n% 100) == 0):
				print(n)
			if(n == test_size):
				break
			length = int(test_line[0])
			test_sub_list = test_line[3:3+length*2]
			test_sub_list = list(map(lambda i: float(i), test_sub_list))
			test_sub_list = same_point_delete(test_sub_list) #same_point_delete
			if(len(test_sub_list)<4): #one or zero point.
				not_passed = not_passed+1
				continue
			while len(test_sub_list) < 20: #so small points
				test_sub_list = add_mid_point(test_sub_list)
			test_sub_list = vectorize_term(test_sub_list, 9)
			test_sub_list = normalize(test_sub_list)
			if(int(len(test_sub_list)/2) != 9):
				print("the length of vector isn't 9")
				not_passed = not_passed+1
				continue
			test_sub_list = data_process2(test_sub_list)
			test_sub_list = random_shake(test_sub_list)
			if(int(len(test_sub_list)/2)!=8):
				print("not 16")
			test_X = np.array(test_sub_list).reshape(-1, 1)
			test_lengths = [2]*int(len(test_sub_list)/2)
			Y = []
			for i in range(0, 10):
				Y.append(model_list[i].score(test_X,test_lengths)) #lengths
			inferences = []
			for i in range(0, 3):
				temp = Y.index(max(Y))
				inferences.append(temp)
				Y[temp] = float('-inf')
			if(num==train_size): #(answer, inference) pair
				t1 = int(test_line[2])
				t2 = inferences[0]
				result[t1][t2] = result[t1][t2]+1
			for i in range(0, 3):
				if(inferences[i] == int(test_line[2])):
					test_answer = test_answer + 1
					break
			n=n+1
		x_test.close()
		acculate = (test_answer / n)*100
		x.append(num)
		y.append(acculate)
	if(num == train_size):
		break
	answer = int(line[2])
	length = int(line[0])

	sub_list = line[3:3+length*2]
	sub_list = list(map(lambda i: float(i), sub_list))
	sub_list = same_point_delete(sub_list) #same_point_delete
	if(len(sub_list)<4): #one or zero point.
		not_passed = not_passed+1
		continue
	while len(sub_list) < 20: #so small points
		sub_list = add_mid_point(sub_list)
	sub_list = vectorize_term(sub_list, 9)
	sub_list = normalize(sub_list)
	if(int(len(sub_list)/2) != 9):
		print("the length of vector isn't 9")
		not_passed = not_passed+1
		coninue
	sub_list = data_process2(sub_list)
	sub_list = random_shake(sub_list)
	if(int(len(sub_list)/2)!=8):
		print("not 8")
	X = np.array(sub_list).reshape(-1, 1)
	lengths = [2] * int(len(sub_list)/2)
	model_list[answer].fit(X, lengths) #lengths
	num = num+1


for i in range(0, 10):
	result_writer.writerow(result[i])

x_train.close()
plt.plot(x, y)
plt.xlabel('the number of train samples')
plt.ylabel('accurate')

plt.title('hmm')
plt.show()

