#!/usr/bin/env python3
import numpy as np
import theano
import theano.tensor as T

def main():
	# Hyperparameter
	SEED=0
	LENGTH=400
	SAMPLESIZE=10
	EMBEDUNIT=48
	OUTPUTSIZE=LENGTH
	RNNUNITSIZE=512
	MAXEPOCH=50000
	MINIBATCHSIZE=SAMPLESIZE//1
	MINIBATCHNUMBER=SAMPLESIZE//MINIBATCHSIZE
	DISTPARAMETER=0.2
	
	# Generating data
	lidata=np.ndarray((SAMPLESIZE,LENGTH+2),dtype=np.int32)
	for i in range(SAMPLESIZE):
		for j in range(LENGTH+2):
			lidata[i][0]=lidata[i][-1]=i
			lidata[i][j-1]=j-2
	lix=lidata[:,:-1]
	lit=lidata[:,1:]
	
	# Input variable
	x=T.imatrix('x')
	t=T.imatrix('t')
	
	# Embedding
	np.random.seed(0)
	embedlong=np.random.uniform(low=-0.2,high=0.2,size=(5000,EMBEDUNIT))
	embed=theano.shared(value=np.asarray(embedlong[:LENGTH,:],dtype=theano.config.floatX))
	np.random.seed(SEED)
	
	# Parameter for GRU layer
	W1xr=theano.shared(value=np.asarray(np.random.uniform(low=-DISTPARAMETER,high=DISTPARAMETER,size=(EMBEDUNIT,RNNUNITSIZE)),dtype=theano.config.floatX),name="W1xr")
	W1xz=theano.shared(value=np.asarray(np.random.uniform(low=-DISTPARAMETER,high=DISTPARAMETER,size=(EMBEDUNIT,RNNUNITSIZE)),dtype=theano.config.floatX),name="W1xz")
	W1xa=theano.shared(value=np.asarray(np.random.uniform(low=-DISTPARAMETER,high=DISTPARAMETER,size=(EMBEDUNIT,RNNUNITSIZE)),dtype=theano.config.floatX),name="W1xa")
	b1r=theano.shared(value=np.asarray(np.random.uniform(low=-DISTPARAMETER,high=DISTPARAMETER,size=(RNNUNITSIZE)),dtype=theano.config.floatX),name="b1r")
	b1z=theano.shared(value=np.asarray(np.random.uniform(low=-DISTPARAMETER,high=DISTPARAMETER,size=(RNNUNITSIZE)),dtype=theano.config.floatX),name="b1z")
	b1a=theano.shared(value=np.asarray(np.random.uniform(low=-DISTPARAMETER,high=DISTPARAMETER,size=(RNNUNITSIZE)),dtype=theano.config.floatX),name="b1a")
	W1hr=theano.shared(value=np.asarray(np.random.uniform(low=-DISTPARAMETER,high=DISTPARAMETER,size=(RNNUNITSIZE,RNNUNITSIZE)),dtype=theano.config.floatX),name="W1hr")
	W1hz=theano.shared(value=np.asarray(np.random.uniform(low=-DISTPARAMETER,high=DISTPARAMETER,size=(RNNUNITSIZE,RNNUNITSIZE)),dtype=theano.config.floatX),name="W1hz")
	W1ha=theano.shared(value=np.asarray(np.random.uniform(low=-DISTPARAMETER,high=DISTPARAMETER,size=(RNNUNITSIZE,RNNUNITSIZE)),dtype=theano.config.floatX),name="W1ha")
	liparameter=[W1xr,W1xz,W1xa,b1r,b1z,b1a,W1hr,W1hz,W1ha]
	vh=theano.shared(value=np.asarray(np.zeros(shape=(MINIBATCHSIZE,RNNUNITSIZE)),dtype=theano.config.floatX),name="vh")
	
	# Parameter for fully connected layer
	W2=theano.shared(value=np.asarray(np.random.uniform(low=-DISTPARAMETER,high=DISTPARAMETER,size=(RNNUNITSIZE,OUTPUTSIZE)),dtype=theano.config.floatX),name="W2")
	b2=theano.shared(value=np.asarray(np.random.uniform(low=-DISTPARAMETER,high=DISTPARAMETER,size=(OUTPUTSIZE)),dtype=theano.config.floatX),name="b2")
	
	# Network calculation
	u=embed[x]
	mu=u.dimshuffle((1,0,2))
	mt=t.dimshuffle((1,0))
	vh,liscanupdate=theano.scan(fn=gruforward,sequences=mu,truncate_gradient=-1,outputs_info=[vh],non_sequences=liparameter)
	mu=T.dot(vh,W2)+b2
	(ud1,ud2,ud3)=mu.shape
	y=T.nnet.softmax(mu.reshape((ud1*ud2,ud3))).reshape((ud1,ud2,ud3))
	(td1,td2)=mt.shape
	tscost=T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(mu.reshape((ud1*ud2,ud3))),mt.reshape((td1*td2,))).reshape((ud1,ud2)))
	
	# Calculating gradient
	liparameter.append(W2)
	liparameter.append(b2)
	ligradient=T.grad(cost=tscost,wrt=liparameter)
	
	# Optimization setting
	liupdate=adam(liparameter,ligradient)
	
	# Generating function
	trainer=theano.function(inputs=[x,t],outputs=[tscost],updates=liupdate)
	my=T.argmax(y.reshape((ud1*ud2,ud3)),axis=1).reshape((ud1,ud2)).dimshuffle((1,0))
	predictor=theano.function(inputs=[x],outputs=[my])
	
	# Learning
	for epoch in range(1,MAXEPOCH+1):
		liindex=np.random.permutation(SAMPLESIZE)
		costvalue=0
		for j in range(MINIBATCHNUMBER):
			start=j*MINIBATCHSIZE
			end=(j+1)*MINIBATCHSIZE
			costvalue+=trainer(lix[liindex[start:end],:],lit[liindex[start:end],:])[0]
		costvalue=costvalue/MINIBATCHNUMBER
		if epoch%10==0:
			lipall,liplast=[],[]
			for j in range(MINIBATCHNUMBER):
				prediction=predictor(lix[j*MINIBATCHSIZE:(j+1)*MINIBATCHSIZE,:])
				lipall+=list(prediction[0].flatten())
				liplast+=list(prediction[0][:,-1].flatten())
			litall=list(lit.flatten())
			litlast=list(lit[:,-1])
			acclast,accall=accuracy(liplast,litlast),accuracy(lipall,litall)
			print("Epoch {0:8d}: Cost={1:9.5f}, ACC(last)={2:7.4f}, ACC(all)={3:7.4f}".format(epoch,float(costvalue),acclast,accall))

def accuracy(lix,liy):
	tp=0
	for x,y in zip(lix,liy):
		if x==y: tp+=1
	return tp/len(lix)

def adam(liparameter,ligradient,a=0.001,b1=0.9,b2=0.999,e=1e-6):
	liupdate=[]
	t=theano.shared(value=np.float32(1),name="t")
	liupdate.append((t,t+1))
	for pc,gc in zip(liparameter,ligradient):
		mc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='mc')
		vc=theano.shared(value=np.zeros(pc.get_value().shape,dtype=theano.config.floatX),name='vc')
		mn=b1*mc+(1-b1)*gc
		vn=b2*vc+(1-b2)*gc**2
		mh=mn/(1-b1**t)
		vh=vn/(1-b2**t)
		pn=pc-(a*mh)/(T.sqrt(vh+e))
		liupdate.append((mc,mn))
		liupdate.append((vc,vn))
		liupdate.append((pc,pn))
	return liupdate

# gruforward
def gruforward(x,vh,W1xr,W1xz,W1xa,b1r,b1z,b1a,W1hr,W1hz,W1ha):
	vz=T.nnet.sigmoid(T.dot(x,W1xr)+b1r+T.dot(vh,W1hr))
	vr=T.nnet.sigmoid(T.dot(x,W1xz)+b1z+T.dot(vh,W1hz))
	vg=T.tanh(T.dot(x,W1xa)+b1a+T.dot(vr*vh,W1ha))
	vh=(1-vz)*vh+vz*vg
	return vh

if __name__ == '__main__':
	main()
