import numpy as np
import tensorflow as tf

WIdef = np.array(np.array([[-.5,-.5,1,1],[-.5,-.5,1,1],[-.5,-.5,1,1],[-.5,-.5,1,1]],dtype = np.float32))

def rect(x):
    return (x + tf.abs(x))/2

def ssn_update(r,I,W,n,k,tau,dt):
    dr = (-r + k * tf.pow(rect(tf.tensordot(W,r,axes = [1,0]) + I),n))/tau
    return r + dt*dr    

def build_response(W,I,r0,n,k,tau,nstep,dt,grad_cutoff = -1):

    R = r0

    res = [R]

    for k in range(nstep):
        res.append(ssn_update(res[-1],I[k],W,n,k,tau,dt))

        if k == grad_cutoff:
            res[-1] = tf.stop_gradient(res[-1])

    return tf.stack(res[1:],axis = 0)

def fit_ssn(I,T,n,k,tau,nstep,dt,n_grad_steps,target_times,grad_cutoff = -1,r0 = tf.zeros(4),WI = WIdef,SGD = False,skip = 1):
    '''
    args:
     I - a list of inputs with shape [nstep,4] the first two indices of the second dim. are E, the last two are I.
     T - a list of target values for the E rates. 
     WI - initial *log* of the absolute value of connection strengths.
     n - power law exponent
     k - recurrent term coefficient
     tau - neuron time constant. Either a float (if all have same tau) or a vector (numpy or tenforflow) of length 4.
     nstep - number of simlation steps to calculate
     dt - time step
     n_grad_steps - number of gradient steps to take
     target_times - a list of when to start enforcing the loss - one for each I/T pair. It is assumed that the loss is calculated till the end of the sim.
     grad_cutoff - when to start calculating BPTT gradients. Default is to calculate BPTT through the entire sim.
     r0 - initial rate values for the network - defaults to 0. Can be a np or tf 4-vector
   
    returns:
     fit values of W
     SSN response trace in response to all the I values.
     final loss
    '''
    #hard signs restricting I and E connections
    print("building network")
    wsign = np.array([[1,1,-1,-1],[1,1,-1,-1],[1,1,-1,-1],[1,1,-1,-1]],dtype = np.float32)

    Wi = tf.Variable(np.float32(WI))

    #the connection strength tensor
    W = wsign * tf.exp(Wi)

    #this builds the response to the input
    RES = [build_response(W,i,r0,n,k,tau,nstep,dt,grad_cutoff) for i in I]

    #calculate the loss. It is just the average squared error in the target times.
    sloss = 0#tf.reduce_sum([tf.reduce_mean(tf.abs(RES[k][target_times[k]:-1] - RES[k][target_times[k] + 1:])) for k in range(len(RES))])

    if SGD:
        loss = [tf.reduce_mean((RES[k][target_times[k]::skip,:1] - tf.expand_dims(T[k],0))**2) for k in range(len(RES))]
        full_loss = tf.reduce_sum(loss)
    else:
        loss = tf.reduce_sum([tf.reduce_mean((RES[k][target_times[k]::skip,:1] - tf.expand_dims(T[k],0))**2) for k in range(len(RES))])
        full_loss = loss


    
    print("calculating training functions")
    #tensorflow stuff
    adam = tf.train.AdamOptimizer()

    if SGD:
        train = [adam.minimize(l + sloss,var_list = [Wi]) for l in loss]
    else:
        train = adam.minimize(loss + sloss,var_list = [Wi])
        
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    l = sess.run(full_loss)
    #training loop
    for k in range(n_grad_steps):
        print("Step {}: {}".format(k,l))

        if SGD:
            k = np.random.permutation(range(len(train)))[0]
            tr,l = sess.run([train[k],full_loss])
            
        else:
            tr,l = sess.run([train,loss])

    #print the resulting weights and final loss
    weights = sess.run(W)
    floss = sess.run(full_loss)
    fres = sess.run(tf.stack(RES,axis = 0))
    
    sess.close()
    return weights, fres, floss
    
    
if __name__=="__main__":

    '''

    The way this is set up is this: you specify the full time course of inputs in a list [I1,I2,I3,...,In] where each Ii is a [ntime,4] array specifying the inputs at each timestep to each neuron for each of n stimulus conditions. Then you specify the targets T = [T1,T2,T3,...,Tn] where each Ti is a 2-vector soepcifying the desired targets for the e-neurons. Finally, you specify test-time which is a list test_time = [t1,t2,t3,...,tn] where each ti is a time step at which you would like to start imposing the loss. See below for an example, see above for function definitions.

    A few more notes:
    
    the neurons are set up as {e1,e2,i1,i2} so teh first two indices are E neurons, and the final two are I neurons. 

    ALso, you can specify "grad cutoff" which only calculats BPTT back to whichever time step is specified by the integer grad_cutoff. This might help if you want to calculate many time steps 

    you can also specify the initial weight matrix with a numpy array (4 by 4) using the argument WI, but note that this is the LOG of the ABSOUTE VALUE of the connection strengths. I use a log parameterization to enforce E/I cell types.
    
    lastly, I have a "skip" option, which tells it how many time steps it can skip between times where it imposes the loss, again, may be helpful when simulating many time steps. 
    '''

    nstep = 200

    i = 1*np.ones([int(nstep/2),4],dtype = np.float32)

    I1 = i*np.array([[1,0,0,0]],dtype = np.float32)
    I1 = np.concatenate([I1,0*I1,0*I1])
    
    I2 = i*np.array([[0,1,0,0]],dtype = np.float32)
    I2 = np.concatenate([I2,0*I2,0*I2])

    I = [I1,I2]
    T = [np.array([2],dtype = np.float32),np.array([1],dtype = np.float32)]

    test_time = [-50,-50]
    dt = .0005
    k = .05
    tau = [.02,.02,.01,.01]
    n = 2.
    n_grad_step = 1000
    skip = 1
    
    W,R,l = fit_ssn(I,T,n,k,tau,nstep,dt,n_grad_step,test_time,grad_cutoff = 0,skip = skip)

    print(R[0,-10:,:2])
