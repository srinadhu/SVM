'''
Written by @srinadhu on Nov 19th.

'''

import smo  #has the code for optimizer
import matplotlib.pyplot as plt #for plotting the decision boundary.
import numpy as np
import math

def Error(X_train,Y_train,alpha,bias,X_test,Y_test,sigma):
	''' Error for the test data'''

	Y_predict =np.zeros(shape=(Y_test.shape[0],1)) #predicted by svm
	Y_t_predict = np.zeros(shape=(Y_train.shape[0],1))
	for i in range(X_test.shape[0]):
		if (smo.predict(X_train,Y_train,alpha,bias,X_test[i,:],sigma) >= 0 ):
			Y_predict[i]=1.0
		else:
			Y_predict[i]=-1.0

	for i in range(X_train.shape[0]):
		if (smo.predict(X_train,Y_train,alpha,bias,X_train[i,:],sigma) >= 0 ):
			Y_t_predict[i]=1.0
		else:
			Y_t_predict[i]=-1.0

	test_error=0.0
	train_error=0.0
	
	for i in range(Y_predict.shape[0]):
		if (Y_predict[i]!=Y_test[i]):
			test_error+=1.0

	
	for i in range(Y_t_predict.shape[0]):
		if (Y_t_predict[i]!=Y_train[i]):
			train_error+=1.0

	return (1-(train_error/Y_train.shape[0]))*100.0,(1-(test_error/Y_test.shape[0]))*100.0

def Matrices(filename):
	'''returns the file input into matrices for both data and labels'''

	labels=[]	

	data=[]

	f=open(filename)  #opening for reading

	for line in f:
		temp=line.split("\t")
		try:
			if (float(temp[0])==0.0):
				labels.append(float(-1.0)) #labels for data
			else:
				labels.append(float(1.0))
		except:
			continue
		temp=temp[1:] #all features.let's do a unit vector normalization.

		for i in range(len(temp)):
			temp[i]=float(temp[i])

		norm=np.linalg.norm(temp) #norm of the input data
		
		for i in range(len(temp)):
			temp[i]= temp[i]/norm   #normalizing it to 1 and 0.
		
		data.append(temp)
	f.close()

	X=np.zeros(shape=(len(data),len(data[i]))) #no of examples and no of features
	Y=np.zeros(shape=(len(data),1)) #for labels

	for i in range(X.shape[0]): #for each row
		X[i,:]=data[i]
		Y[i,:]=labels[i]

	return X,Y

def alphas(alpha,C=2.0):
	'''returns the number of ranges of them'''
	a=0
	b=0
	c=0
	for i in range(alpha.shape[0]):
		if (alpha[i]==0.0):
			a+=1
		elif (alpha[i]==C):
			b+=1
		else:
			c+=1
	return a,b,c

def plot(Train_error,Test_error,alphas_cc,alphas_mc,alphas_rm,Sigmas):
	'''returns the plots'''
	
	plt.plot(Sigmas,Train_error,color='r')
	plt.plot(Sigmas,Test_error,color='b')
	plt.xlabel("Degree")
	plt.ylabel("Train & Test Accuracy")
	plt.title("Classification vs Degree of Polynomial Kernel. \n(r-train\nb-test)\n")
	plt.savefig("./class_error.png", bbox_inches='tight')
	plt.clf()


	plt.plot(Sigmas,alphas_cc,color='r')
	plt.plot(Sigmas,alphas_mc,color='b')
	plt.plot(Sigmas,alphas_rm,color='g')
	plt.xlabel("Degree")
	plt.ylabel("Support Vectors")
	plt.title("Support Vectors vs Degree. \n(r-Lagrange multipler value 0\nb-Lagrange multipler value C\ng-Lagrange multipler value b/w 0 and C)\n")
	plt.savefig("./support_vectors.png", bbox_inches='tight')
	plt.clf()

X_train,Y_train=Matrices("train")
X_test,Y_test=Matrices("test")

Train_error=[]
Test_error=[]
Sigmas=[]
alphas_cc=[] #correctly classfied
alphas_mc=[] #mis classified
alphas_rm=[] #middle ones

sigma=0

while(sigma<5):
	sigma+=1
	Sigmas.append(sigma)
	print "called SMO"
	alpha,bias= smo.SMO(X_train,Y_train,1.0,math.pow(10,-3),2,sigma)  #with varying sigma
	tr_err,tst_err= Error(X_train,Y_train,alpha,bias,X_test,Y_test,sigma)
	Train_error.append(tr_err)
	Test_error.append(tst_err)
	a,b,c = alphas(alpha)
	print alpha
	print a,b,c
	print tr_err
	print tst_err
	alphas_cc.append(a)
	alphas_mc.append(b)
	alphas_rm.append(c)
	print "one call done"
plot(Train_error,Test_error,alphas_cc,alphas_mc,alphas_rm,Sigmas)
