'''
Written by @srinadhu on Nov 19th.

reference: http://cs229.stanford.edu/materials/smo.pdf

'''


import numpy as np #for dealing with matrices.
import math #for exponential and power function
import random #for random number generation



def gaussian_kernel(x1,x2,sigma=1.0):
	'''returns the dot product in infinite dimensional space'''

	norm= np.linalg.norm(np.subtract(x1,x2)) #norm 

	return math.exp(-(norm**2)/(2*(sigma**2))) #returning the final dot product.


def polynomial_kernel(x1,x2,degree=1):
	'''returns the dot in trnasformed polynomial space'''

	dot_prdt= np.dot (np.transpose(x1),x2) #give the dot product 

	return (dot_prdt+1)**degree  #returning the final dot product.


def predict(X,Y,alpha,b,x):
	'''predict the value for a new data point'''

	result=0

	for i in range(X.shape[0]):
		result+=(alpha[i]*Y[i]*gaussian_kernel(X[i,:] , x));

	result+=b

	return result


def SMO(X,Y,C=0.5,tol=math.power(10,-5),max_passes=1000):
	''' X has input data matrix. Y has the class labels. C is regularization parameter. tol is numerical tolerance. max_passes is max # of times to iterate wihtout changing alpha's

        Return Alpha and b.'''

    alpha=np.zeros(shape=(X.shape[0],1)); # each alpha[i] for every example.
	b=0
	
	passes=0

	E=np.zeros(shape=(X.shape[0],1)) #will be used in the loop
	alpha_old=alpha #will be used in the loop

	while(passes < max_passes):
		num_changed_alphas=0
		
		for i in range(len(X.shape[0])): #for every example
			E[i]=(predict(X,Y,alpha,b,X[i,:])-Y[i])
 
    		if ( (Y[i]*E[i] < -tol && alpha[i] < C ) || ( Y[i]*E[i] > tol && alpha[i] >0 ) ):
				j=i
				while(j!=i):
					j=random.randrange(X.shape[0]) #get any other data point other than i

				E[j] = (predict(X,Y,alpha,b,X[j,:])-Y[j]) #for other data point

				alpha_old[i]=alpha[i]
				alpha_old[j]=alpha[j]

				#computing L and h values

				if (Y[i]!=Y[j]):
					L=max(0,alpha[j]-alpha[i])
					H=min(C,C+alpha[j]-alpha[i])
				else:
					L=max(0,alpha[i]+alpha[j]-C)
					H=min(C,alpha[i]+alpha[j])

				if (L==H):
					continue
				eta = 2 *(gaussian_kernel(X[i,:],X[j,:])) - (gaussian_kernel(X[i,:],X[i,:]) - (gaussian_kernel(X[j,:],X[j,:])

				if (eta >=0):
					continue
	
				#clipping
				if (alpha[j] > H):
					alpha[j]=H
				elif (alpha[j]<L):
					alpha[j]=L
				else
					pass

				if (abs(alpha[j]-alpha_old[j]) < tol):
					continue
			
				alpha[i] += (Y[i]*Y[j]*(alpha_old[j] - alpha[j])) #both alphas are updated

				b1= b-E[i]-(Y[i]*(alpha[i]-alpha_old[i])*(gaussian_kernel(X[i,:],X[i,:])))-(Y[j]*(alpha[j]-alpha_old[j])*(gaussian_kernel(X[i,:],X[j,:])))
				b2= b-E[j]-(Y[i]*(alpha[i]-alpha_old[i])*(gaussian_kernel(X[i,:],X[j,:])))-(Y[j]*(alpha[j]-alpha_old[j])*(gaussian_kernel(X[j,:],X[j,:])))

				if (alpha[i] > 0 && alpha[i]<C):
					b=b1
				elif (alpha[j] > 0 && alpha[j] <C):
					b=b2
				else:
					b=(b1+b2)/2.0
				
				num_changed_alphas+=1
			#ended if
		#ended for
		if (num_changed_alphas == 0):
			passes+=1
		else:
			passes=0
	#end while

	return alpha,b   #returning the lagrange multipliers and bias.
					
