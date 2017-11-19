'''
Written by @srinadhu on Nov 19th.

'''

import smo  #has the code for optimizer
import matplotlib.pyplot as plt #for plotting the decision boundary.
import numpy as np

def Error(X_train,Y_train,alpha,bias,X_test,Y_test):
	''' Error for the test data'''

	Y_predict =np.zeros(shape=(Y_test.shape[0],1)) #predicted by svm

	for i in range(X_test.shape[0]):
		if (smo.predict(X_train,Y_train,alpha,bias,X_test[i,:]) >= 0 ):
			Y_predict[i]=1.0
		else:
			Y_predict[i]=0.0

	test_error=0.0
	train_error=0.0

	for i in range(Y_predict.shape[0]):
		if (Y_predict[i]!=Y_test[i]):
			test_error+=1.0

	return test_error/Y_test.shape[0]

def Matrices(filename):
	'''returns the file input into matrices for both data and labels'''

	labels=[]	

	data=[]

	f=open(filename)  #opening for reading

	for line in f:
		temp=line.split("\t")
		try:
			labels.append(float(temp[0])) #labels for data
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

X_train,Y_train=Matrices("train")
print "called SMO"
alpha,bias= smo.SMO(X_train,Y_train)

print alpha
print bias

X_test,Y_test=Matrices("test")
print "done"
print Error(X_train,Y_train,alpha,bias,X_test,Y_test)
