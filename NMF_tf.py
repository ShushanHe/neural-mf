# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:01:57 2019

@author: shush
"""
################################################################
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import time
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser('NMF_tf')
parser.add_argument('--nNodes', type=int, default=128)
parser.add_argument('-T','--time_window', type=int, default=20)
parser.add_argument('-bs','--batch_size', type=int, default=300)
#parser.add_argument('--Keywords', type=str, default='HE_128_512')
#parser.add_argument('--filenumber', type=int, default=1)
parser.add_argument('--train_size', type=int, default=9000)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--patience', type=int, default=1000)

#args = parser.parse_args()
args, unknown = parser.parse_known_args()


################################################################
test_size=80
tcut=20
ncut=20
n_layers=ncut+1
dt=tcut/ncut

lambda1=0.001#regularization rate for weight_A
REGULARIZATION_RATE=0.0001#regularization rate for other parameters
learning_rate=0.001

################################Forward Process##################################################
def get_weight_variable(shape, regularizer,name='weights'):
    initializer=tf.random_normal_initializer(mean=0.0,stddev=0.00001)
    weights = tf.compat.v1.get_variable(name, shape, dtype=tf.float32,initializer=initializer)
    if regularizer != None: tf.compat.v1.add_to_collection('LOSSES_COLLECTION', regularizer(weights))
    return weights 

def binary_cross_entropy(true_y,pred_y):
    cross_loss=tf.reduce_mean(-true_y*tf.log(tf.clip_by_value(pred_y,1e-10,1.0))-(1-true_y)*tf.log(tf.clip_by_value((1-pred_y),1e-10,1.0)))
    return cross_loss

def mae(true_y,pred_y):
    #pred_y_act=tf.clip_by_value(pred_y,1e-10,1.0-(1e-10))
    mae=tf.reduce_mean(tf.abs(true_y-pred_y))
    return mae
#%%
    
def inf_erro_CNN(input_tensor,nNodes,regularizer):
    #the inner neural network to learn h 
    #note that： in each timestep, we share the same parameters.
    INPUT_NODE=nNodes
    OUTPUT_NODE = nNodes
    LAYER1_NODE= nNodes
    LAYER2_NODE=nNodes
    with tf.compat.v1.variable_scope('layer1',reuse=tf.compat.v1.AUTO_REUSE):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.compat.v1.get_variable("biases", [LAYER1_NODE], dtype=tf.float32,
                                           initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.elu(tf.nn.bias_add(tf.matmul(input_tensor, weights), biases))        
    with tf.compat.v1.variable_scope('layer2',reuse=tf.compat.v1.AUTO_REUSE):    
        weights = get_weight_variable([LAYER1_NODE, LAYER2_NODE], regularizer)
        biases = tf.compat.v1.get_variable("biases", [LAYER2_NODE], dtype=tf.float32,
                                           initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.elu(tf.nn.bias_add(tf.matmul(layer1, weights), biases))           
    with tf.compat.v1.variable_scope('layer3',reuse=tf.compat.v1.AUTO_REUSE):    
        weights = get_weight_variable([LAYER2_NODE, OUTPUT_NODE], regularizer)
        biases = tf.compat.v1.get_variable("biases", [OUTPUT_NODE], dtype=tf.float32,
                                           initializer=tf.constant_initializer(0.0))
        layer3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer2, weights), biases))           
    return layer3 


def inf_LSTM_MTM(xinput,nNodes,pre_c,pre_h,regularizer,forget_bias=1.0):
    # define the LSTM cell
    num_hidden=nNodes
    with tf.compat.v1.variable_scope('com',reuse=tf.compat.v1.AUTO_REUSE):
        betaLSTM=tf.compat.v1.get_variable('betaLSTM',shape=[1],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        inputComb=betaLSTM*xinput+pre_h        
    with tf.compat.v1.variable_scope('f',reuse=tf.compat.v1.AUTO_REUSE):
        weightf=get_weight_variable([nNodes, num_hidden], regularizer,name='fw')
        biasesf = tf.compat.v1.get_variable("biases", [num_hidden], dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.0))
        f = tf.nn.bias_add(tf.matmul(inputComb, weightf), biasesf)        
    with tf.compat.v1.variable_scope('i',reuse=tf.compat.v1.AUTO_REUSE):
        weighti=get_weight_variable([nNodes, num_hidden], regularizer,name='iw')
        biasesi = tf.compat.v1.get_variable("biases", [num_hidden], dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.0))
        i = tf.nn.bias_add(tf.matmul(inputComb, weighti), biasesi)       
    with tf.compat.v1.variable_scope('o',reuse=tf.compat.v1.AUTO_REUSE):    
        weighto=get_weight_variable([nNodes, num_hidden], regularizer,name='ow')
        biaseso = tf.compat.v1.get_variable("biases", [num_hidden], dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.0))
        o = tf.nn.bias_add(tf.matmul(inputComb, weighto), biaseso) 
    with tf.compat.v1.variable_scope('j',reuse=tf.compat.v1.AUTO_REUSE):    
        weightj=get_weight_variable([nNodes, num_hidden], regularizer,name='jw')
        biasesj = tf.compat.v1.get_variable("biases", [num_hidden], dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.0))
        j = tf.nn.bias_add(tf.matmul(inputComb, weightj), biasesj)  
    cur_c=tf.nn.sigmoid(f + forget_bias) * pre_c +tf.nn.sigmoid(i) * tf.nn.tanh(j)
    cur_h=tf.nn.sigmoid(o) * tf.nn.tanh(cur_c)  
    return cur_c,cur_h   
 
def inf_full_LSTM(input_layer,nNodes,supp_A,layer_n,lambda1,regularizer,num_hidden,pre_c,pre_h,forget_bias=1.0):
    # define the full block at time t=layer_n
    #lambda1 is the coefficient of regularizer loss
    #if lambda1=None, we will not count the regularizer loss. eg. in validation and test
    in_dimension=nNodes
    out_dimension=nNodes
    AdjMatrix_tensor=tf.constant(supp_A,dtype=tf.float32)
    
    with tf.compat.v1.variable_scope('layer',reuse=tf.compat.v1.AUTO_REUSE):
        
        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.00001)
        #initializer=tf.contrib.layers.xavier_initializer()
        weightA=tf.compat.v1.get_variable('weightA',shape=[in_dimension,out_dimension],dtype=tf.float32,
                                          initializer=initializer,
                                          constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
        weightA_sparse=tf.multiply(weightA,AdjMatrix_tensor)
        if layer_n==1 and lambda1!=None:
            tf.compat.v1.add_to_collection('LOSSES_COLLECTION',
                                           tf.contrib.layers.l1_regularizer(lambda1)(weightA_sparse))
        temp1=tf.matmul(input_layer,weightA_sparse)
        temp1_reshape=tf.reshape(temp1,[-1,1,nNodes])
        diag=tf.matrix_diag(input_layer)
        temp2_reshape=tf.matmul(temp1_reshape,diag)
        temp2=tf.reshape(temp2_reshape,shape=[-1,nNodes])
        phi=input_layer+dt*(temp1-temp2)
        
        cur_c,cur_h=inf_LSTM_MTM(input_layer,nNodes,pre_c,pre_h,regularizer,forget_bias=forget_bias)           
        
        betaCNN=tf.compat.v1.get_variable('betaCNN',shape=[1],dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.1)) 
        new_inp=input_layer+betaCNN*cur_h

        error=inf_erro_CNN(new_inp,nNodes,regularizer)             
        
        layer=tf.clip_by_value(phi-dt*error,0,1)

    return layer,cur_c,cur_h


#%%
class NMF:
    def __init__(self,Keywords,TRAINING_STEPS,nNodes,nEdges):
        self.Keywords=Keywords+'_'+str(nNodes)+'_'+str(nEdges)
        self.max_training_steps=TRAINING_STEPS
        self.nNodes=nNodes
        self.nEdges=nEdges
        
        # note that: "cascades-train-support-1.npy" is a matrix of size (nNodes,nNodes) with element x_ij,
        # where x_ij=1 if nodes i and j are in the same cascade and t_i<t_j, x_ij=0 otherwise.
        # Clearly, it only uses the infection time log of cascades data.
        # For our model, this matrix is not always necessary. Only when the network is very sparsity
        # or the cascades data only covers few nodes, the supp_A will improve the robustness of our model.
        # But most of time, it only change the diagonal elements of weight matrix A to 0 which is trivial. 
        self.support_A_file=self.Keywords+'/Datasets/cascades-train-support-1.npy'  
        self.supp_A=np.load(self.support_A_file).astype('float32')
                
        self.loss_savefile=self.Keywords+'/Results/Loss-NMF-'+self.Keywords+'.pdf'
        self.weight_savefile=self.Keywords+'/Results/WeightBias-NMF-'+self.Keywords+'.npy'
        self.output_savefile=self.Keywords+'/Results/OutProb-NMF-'+self.Keywords+'.npy'
        self.modeldir=self.Keywords+'/Models'
        self.MODEL_NAME=self.modeldir+'/model-NMF-'+self.Keywords+'.ckpt'        
        self.BATCH_SIZE=50
        self.early_stopping=50#if the validation mae does not change for 20 trains, then break the training
               
    def train(self,TrainData,ValiData,losstype='cs',forget_bias=1.0):
        start_time=time.time()
        with tf.Graph().as_default():
            x=tf.compat.v1.placeholder(tf.float32,shape=[None,self.nNodes],name='x-input')
            y_=tf.compat.v1.placeholder(tf.float32,shape=[None,None,self.nNodes],name='y-input')
            
            regularizer = tf.contrib.layers.l1_regularizer(REGULARIZATION_RATE)
            cur_layer=x
            
            num_hidden=self.nNodes
            first_dim, last_dim = tf.unstack(tf.shape(cur_layer))
            pre_c=tf.zeros(shape=[first_dim,num_hidden],dtype=tf.float32)
            pre_h=tf.zeros(shape=[first_dim,num_hidden],dtype=tf.float32)
            
            lossmain=0
            totalmae=0
            for layer_n in range(1,n_layers):               
                cur_layer,pre_c,pre_h=inf_full_LSTM(cur_layer,self.nNodes,self.supp_A,layer_n,lambda1,regularizer,num_hidden,pre_c,pre_h) 
                if losstype=='MSE':
                    loss_temp=tf.reduce_mean(tf.keras.losses.MSE(y_[:,layer_n,:],cur_layer)) 
                if losstype=='cs':
                    loss_temp=binary_cross_entropy(y_[:,layer_n,:],cur_layer)   
                totalmae+=mae(y_[:,layer_n,:],cur_layer)                        
                lossmain=lossmain+loss_temp        
            
            global_step=tf.compat.v1.Variable(0,trainable=False)
            loss=lossmain+tf.add_n(tf.compat.v1.get_collection('LOSSES_COLLECTION'))
    
            original_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=2.0)     
            train_step = optimizer.minimize(loss,global_step)        
            saver=tf.compat.v1.train.Saver()
            
            train_size=TrainData.shape[0]
            epoch_dim=int(train_size/self.BATCH_SIZE)
            display_step=min(200,epoch_dim)
            
            best_val_loss = 1000
            best_val_mae = 1000
            
            with tf.compat.v1.Session(config=config) as sess:
                tf.compat.v1.global_variables_initializer().run()
                loss_list=[] 
                for j in range(self.max_training_steps):
                    start=(j*self.BATCH_SIZE)%TrainData.shape[0]
                    end=min(start+self.BATCH_SIZE,TrainData.shape[0])
                    train_feed={x:TrainData[start:end,0,:],y_:TrainData[start:end,:,:]}
                    #idx = np.random.randint(TrainData.shape[0], size=BATCH_SIZE)
                    #train_feed={x:TrainData[idx,0,:],y_:TrainData[idx,:,:]}  
                    sess.run(train_step,feed_dict=train_feed)
                    loss_value,step=sess.run([loss,global_step],feed_dict=train_feed)
                    loss_list.append(loss_value)
                    if j%display_step==0:
                        print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                        val_dict={x:ValiData[:,0,:],y_:ValiData}
                        loss_val=sess.run(loss,feed_dict=val_dict)
                        mae_val=sess.run(totalmae,feed_dict=val_dict)

                        if mae_val<best_val_mae:
                            best_val_loss=loss_val
                            best_val_mae=np.mean(mae_val)
                            patience=self.early_stopping                            
                            saver.save(sess,self.MODEL_NAME,global_step=global_step)  
                        patience-=1
                        if not patience:
                            break                    
                #total_loss=sess.run(loss,feed_dict={x:TrainData[:,0,:],y_:TrainData}) 
            print("Valid Loss",best_val_loss)
            print("Valid mae",best_val_mae)
            #print("After %d training step(s), total loss on training data is %g." % (j, total_loss))
            
            run_time=time.time()-start_time
            print("runtime= ",run_time)
            plt.plot(loss_list)  
            plt.xlabel('Steps')
            if losstype=='MSE':
                plt.ylabel('MSE Loss')
            if losstype=='cs':
                plt.ylabel('Cross Entropy Loss')
            plt.savefig(self.loss_savefile)
            plt.close()              
        return run_time,j
    ###################################Evaluate Process########################################
    def test(self,TestData,losstype='cs',forget_bias=1.0):

        with tf.Graph().as_default():
            x=tf.compat.v1.placeholder(tf.float32,shape=[None,self.nNodes],name='x-input')
            y_=tf.compat.v1.placeholder(tf.float32,shape=[None,None,self.nNodes],name='y-input')
            test_feed = {x: TestData[:,0,:], y_: TestData}    	
            cur_layer=x
            
            num_hidden=self.nNodes
            first_dim, last_dim = tf.unstack(tf.shape(cur_layer))
            pre_c=tf.zeros(shape=[first_dim,num_hidden],dtype=tf.float32)
            pre_h=tf.zeros(shape=[first_dim,num_hidden],dtype=tf.float32)        
            
            loss=0
            output={}
            output[0]=cur_layer
            for layer_n in range(1,n_layers):
                cur_layer,pre_c,pre_h=inf_full_LSTM(cur_layer,self.nNodes,self.supp_A,layer_n,lambda1,None,num_hidden,pre_c,pre_h,forget_bias=forget_bias)
                output[layer_n]=cur_layer
                if losstype=='MSE':
                    loss_temp=tf.reduce_mean(tf.keras.losses.MSE(y_[:,layer_n,:],cur_layer))  
                if losstype=='cs':
                    loss_temp=binary_cross_entropy(y_[:,layer_n,:],cur_layer)
                loss=loss+loss_temp            
            saver=tf.compat.v1.train.Saver()
            with tf.compat.v1.Session() as sess:
                 ckpt = tf.train.get_checkpoint_state(self.modeldir)
                 if ckpt and ckpt.model_checkpoint_path:                 
                     saver.restore(sess, ckpt.model_checkpoint_path)
                     loss_test=sess.run(loss,feed_dict=test_feed)             
                     LayOutput=sess.run(output,feed_dict=test_feed)                    
                     np.save(self.output_savefile,LayOutput)
                     
                     saver.export_meta_graph(ckpt.model_checkpoint_path+'.json',as_text=True)
                     reader=tf.compat.v1.train.NewCheckpointReader(ckpt.model_checkpoint_path)
                     all_variables=reader.get_variable_to_shape_map()
                     weightbias_tensor={}
                     for variable_name in all_variables:
                         weightbias_tensor[variable_name]=reader.get_tensor(variable_name)                
                     np.save(self.weight_savefile,weightbias_tensor)                    
                     print(("The test loss using average model is %g" %(loss_test)))
                 else:
                     print('No checkpoint file found')
        return
    
def main(argv=None):   
    TRAINING_STEPS=15000
    Keyword='HR'
    nNodes=128
       
    nEdges=int(4*nNodes)
    Keywords=Keyword+'_'+str(nNodes)+'_'+str(nEdges)
    Model=NMF(Keyword,TRAINING_STEPS,nNodes,nEdges)
    
    TrainData=np.load(Keywords+'/Datasets/cascades-train-1.npy')
    TestData=np.load(Keywords+'/Datasets/cascades-test-1.npy')
    ValiData=TestData[0:20,:,:]    
    runtime,j=Model.train(TrainData,ValiData)
    Model.test(TestData[20::,:,:])
    
    f=open('runtime_log.txt','a')
    f.write('nNodes: '+str(nNodes)+'\n')
    f.write('runtime: '+str(runtime)+'\n')
    f.write('trainsteps: '+str(j)+'\n')
        
if __name__=='__main__':
  tf.compat.v1.app.run()
