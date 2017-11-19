'''
dataset taken from: shttps://archive.ics.uci.edu/ml/datasets/HIGGS

53% positive 
47% negative
'''
import random #for shuffling the data needed

p=636
n=564

pos=[]
neg=[]

f=open("HIGGS.csv")

for line in f:
	temp=line.split(",")
	
	if (float(temp[0]) == 1):
		if (p>0):
			pos.append(temp)
			p-=1
		else:
			continue;
	else:
		if (n>0):
			neg.append(temp)
			n-=1
		else:
			continue;

f.close()

random.shuffle(pos)
random.shuffle(neg)


train=[]
test=[]

for i in range(636):
	if i<530:
		train.append(pos[i])
	else:
		test.append(pos[i])

for i in range(564):
	if i<470:
		train.append(neg[i])
	else:
		test.append(neg[i])

f=open("train",'w')

for i in range(len(train)):
	for j in range(len(train[i])):
		f.write(train[i][j]+"\t")
	f.write("\n")
f.close()

f=open("test",'w')

for i in range(len(test)):
	for j in range(len(test[i])):
		f.write(test[i][j]+"\t")
	f.write("\n")
f.close()
