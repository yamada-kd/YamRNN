#!/usr/bin/env python3
import tensorflow as tf
tf.random.set_seed(0)
from tensorflow.python.keras import activations,constraints,initializers,regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import recurrent
import numpy as np

def main():
	# Sample dataset
	tx=[[[1.1],[2.2],[3.0],[4.0]],[[2.0],[3.0],[4.0],[1.0]],[[2.0],[3.0],[4.0]]]
	tx=tf.keras.preprocessing.sequence.pad_sequences(tx,padding="post",dtype=np.float32)
	tt=[1,2,3]
	tt=tf.convert_to_tensor(tt)
	
	# Network
	model=Network()
	cce=tf.keras.losses.SparseCategoricalCrossentropy()
	acc=tf.keras.metrics.SparseCategoricalAccuracy()
	optimizer=tf.keras.optimizers.Adam()
	
	# Decorator
	@tf.function
	def inference(tx,tt):
		with tf.GradientTape() as tape:
			ty=model.call(tx)
			traincost=cce(tt,ty)
		gradient=tape.gradient(traincost,model.trainable_variables)
		optimizer.apply_gradients(zip(gradient,model.trainable_variables))
		trainacc=acc(tt,ty)
		return traincost,trainacc
	
	# Inference
	for epoch in range(1,1000+1):
		traincost,trainacc=inference(tx,tt)
		if epoch%10==0:
			print(epoch,traincost,trainacc)
			ty=model.call(tx)
			print(ty,tt)

class Network(tf.keras.layers.Layer):
	def __init__(self):
		super(Network,self).__init__()
		self.lstm=tf.keras.layers.RNN(ENCell(5))
		self.fc=tf.keras.layers.Dense(4,activation="softmax")
	
	def call(self,tx):
		ty=self.lstm(tx)
		ty=self.fc(ty)
		return ty

class ENCell(tf.keras.layers.AbstractRNNCell,recurrent.DropoutRNNCellMixin):
	def __init__(self,units,activation="tanh",use_bias=True,kernel_initializer="glorot_uniform",recurrent_initializer="orthogonal",bias_initializer="zeros",kernel_regularizer=None,recurrent_regularizer=None,bias_regularizer=None,kernel_constraint=None,recurrent_constraint=None,bias_constraint=None,dropout=0.,recurrent_dropout=0.,**kwargs):
		super(ENCell,self).__init__(**kwargs)
		self.units=units
		self.activation=activations.get(activation)
		self.use_bias=use_bias
		self.kernel_initializer=initializers.get(kernel_initializer)
		self.recurrent_initializer=initializers.get(recurrent_initializer)
		self.bias_initializer=initializers.get(bias_initializer)
		self.kernel_regularizer=regularizers.get(kernel_regularizer)
		self.recurrent_regularizer=regularizers.get(recurrent_regularizer)
		self.bias_regularizer=regularizers.get(bias_regularizer)
		self.kernel_constraint=constraints.get(kernel_constraint)
		self.recurrent_constraint=constraints.get(recurrent_constraint)
		self.bias_constraint=constraints.get(bias_constraint)
		self.dropout=min(1.,max(0.,dropout))
		self.recurrent_dropout=min(1.,max(0.,recurrent_dropout))
	
	@property
	def state_size(self):
		return self.units
	
	def build(self,input_shape):
		input_dim=input_shape[-1]
		self.kernel=self.add_weight(shape=(input_dim,self.units),name="kernel",initializer=self.kernel_initializer,regularizer=self.kernel_regularizer,constraint=self.kernel_constraint)
		self.recurrent_kernel=self.add_weight(shape=(self.units,self.units),name="recurrent_kernel",initializer=self.recurrent_initializer,regularizer=self.recurrent_regularizer,constraint=self.recurrent_constraint)
		if self.use_bias:
			self.bias=self.add_weight(shape=(self.units,),name="bias",initializer=self.bias_initializer,regularizer=self.bias_regularizer,constraint=self.bias_constraint)
		else:
			self.bias=None
		self.built=True
	
	def call(self,inputs,states,training=None):
		vh=states[0]
		
		dp_mask=self.get_dropout_mask_for_cell(inputs,training,count=1)
		rec_dp_mask=self.get_recurrent_dropout_mask_for_cell(vh,training,count=1)
		
		if 0.<self.dropout<1.:
			input1=inputs*dp_mask[0]
		else:
			input1=inputs
		
		if 0.<self.recurrent_dropout<1.:
			vh1=vh*rec_dp_mask[0]
		else:
			vh1=vh
		
		p11=K.dot(input1,self.kernel)
		if self.use_bias:
			p11=K.bias_add(p11,self.bias)
		
		vh=self.activation(p11+K.dot(vh1,self.recurrent_kernel))
		return vh,[vh]
	
	def get_config(self):
		config={"units":self.units,"activation":activations.serialize(self.activation),"use_bias":self.use_bias,"kernel_initializer":initializers.serialize(self.kernel_initializer),"recurrent_initializer":initializers.serialize(self.recurrent_initializer),"bias_initializer":initializers.serialize(self.bias_initializer),"kernel_regularizer":regularizers.serialize(self.kernel_regularizer),"recurrent_regularizer":regularizers.serialize(self.recurrent_regularizer),"bias_regularizer":regularizers.serialize(self.bias_regularizer),"kernel_constraint":constraints.serialize(self.kernel_constraint),"recurrent_constraint":constraints.serialize(self.recurrent_constraint),"bias_constraint":constraints.serialize(self.bias_constraint),"dropout":self.dropout,"recurrent_dropout":self.recurrent_dropout}
		base_config=super(ENCell,self).get_config()
		return dict(list(base_config.items())+list(config.items()))

if __name__ == "__main__":
	main()
