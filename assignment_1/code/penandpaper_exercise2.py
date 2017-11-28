from __future__ import division
import numpy as np
from copy import deepcopy



def RELUderivative(M):
  N=deepcopy(M)
  N[N>0]  = 1
  N[N<=0] = 0
  return N


#if __name__ == '__main__':
# 1. forward input and output of layers###################################################
x = np.matrix('[0.75,0.8;0.2,0.05;-0.75,0.8;0.2,-0.05]')
w = np.matrix('[0.6,0.7,0;0.01,0.43,0.88]')
s = x*w
# https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
z = np.maximum(s, 0)#np.multiply(s,s>0)
w_out = np.matrix('[0.02;0.03;0.09]')
s_out = z*w_out


# 2. compute Loss#########################################################################
y = np.matrix('[1;1;-1;-1]')
L = 1/2*np.power((s_out - y),2)
np.sum(L)


# 3. Compute the error signal at the output################################################
# https://kawahara.ca/what-is-the-derivative-of-relu/
sigma_out = np.multiply((s_out - y), RELUderivative(s_out))#*f_3'(s_out)#->[1;1;1;1]
np.sum(sigma_out)


# 4. Propagate sigma_out backwards  to compute sigma_j at hidden units#####################
sigma_j = np.multiply(sigma_out*np.transpose(w_out),RELUderivative(s))


# 5.  Compute gradients (i.e. dW, dw) using these and update weights using#################
#     gradient descent. (Take learning rate as 0.5)
#     Hint: W(t+1) = W(t) - alpha * dW
dW_out    = np.multiply(sigma_out,z)
dW_out    = np.transpose(sigma_out)*z
dW_out    = sigma_out*np.transpose(z)
dW_out    = np.multiply(sigma_out,z).mean(axis=0)

W_out_tp1 = w_out - 0.5*np.transpose(dW_out)

dW_1      = np.transpose(sigma_j)*x
dW_1      = np.multiply(sigma_j,x).mean(axis=0)
W_1_tp1   = w - 0.5*np.transpose(dW_1)





def timing_results():
	import timeit
	#TIMING RESULTS
	x = np.matrix(np.random.random((5000, 5000)) - 0.5)
	print("max method:")
	timeit.timeit('np.maximum(x, 0)', setup='import numpy as np; x=np.matrix(np.random.random((500, 500)) - 0.5)', number=10000)#13.602573016998576
	#%timeit -n10 np.maximum(x, 0)# 239 ms per loop

	print("multiplication method:")
	timeit.timeit('x*(x>0)', setup='import numpy as np; x=np.matrix(np.random.random((500, 500)) - 0.5)', number=10000)#77.1596870599933
	#timeit.timeit('x*(x>0)', number=10000)
	#%timeit -n10 x * (x > 0)# 145 ms per loop

	print("multiplication method (numpy):")
	timeit.timeit('np.multiply(x,x>0)', setup='import numpy as np; x=np.matrix(np.random.random((500, 500)) - 0.5)', number=10000)#15.292496291000134
	#%timeit -n10 np.multiply(x,x>0)# 145 ms per loop

	print("abs method:")
	timeit.timeit('(abs(x) + x) / 2', setup='import numpy as np; x=np.matrix(np.random.random((500, 500)) - 0.5)', number=10000)#9.588070144000085
	#%timeit -n10 (abs(x) + x) / 2# 288 ms per loop
