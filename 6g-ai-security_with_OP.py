# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 23:25:11 2023

@author: Ammar.Abasi
"""

#!/usr/bin/env python
# coding: utf-8

# # 6G and Artificial Intelligence With Security Problems
# ## 6G solutions with Adversarial Machine Learning Attacks: Millimeter Wave Beam Prediction Use-Case
# 



import tensorflow as tf
from scipy.io import loadmat, savemat
import numpy as np
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
import pickle 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import os



loss_object = tf.keras.losses.MeanSquaredError()

def fgsm(model, input_instance, label, n_BS,n_beams, epsilon =0.01):
    tensor_input_instance = tf.convert_to_tensor(input_instance, dtype=tf.float32)
    adv_x = input_instance
    for idx in range(0, n_BS*n_beams-2, n_beams):
        with tf.GradientTape() as tape:
            tmp_label = label[:, idx:idx + n_beams]
            tape.watch(tensor_input_instance)
            prediction = model(tensor_input_instance)
            loss = loss_object(tmp_label, prediction)
            gradient = tape.gradient(loss, tensor_input_instance)
            signed_grad = tf.sign(gradient)
            adv_x = adv_x + eps * signed_grad
    return adv_x





# Reading input and output sets generated from MATLAB
# with DeepMIMO generator http://www.deepmimo.net
In_set_file=loadmat('DLCB_dataset/DLCB_input.mat')
Out_set_file=loadmat('DLCB_dataset/DLCB_output.mat')

In_set=In_set_file['DL_input']
Out_set=Out_set_file['DL_output']

# Parameter initialization
num_user_tot=In_set.shape[0]

DL_size_ratio = .8
count=0
num_tot_TX=4
num_beams=64







def train(In_train, Out_train, In_test, Out_test,
          nb_epoch, batch_size,dr,
          num_hidden_layers, nodes_per_layer,
          loss_fn,n_BS,n_beams):
    
    in_shp = list(In_train.shape[1:])
#94   1   2 128
    AP_models = []
    for idx in range(0, n_BS*n_beams-2, n_beams):
        idx_str = str(idx / n_beams + 1)
        act_func = 'relu'
        model = Sequential()
        model.add(Dense(94, input_dim=in_shp[0], activation=act_func))
        model.add(Dense(94, activation=act_func))
        model.add(Dense(94, activation=act_func))
        model.add(Dense(94, activation=act_func))
        model.add(Dense(n_beams, activation=act_func))
        model.compile(loss=loss_fn, optimizer='ADAM', metrics=['mean_squared_error'])
        
        history = model.fit(In_train,
                            Out_train[:, idx:idx + n_beams],
                            batch_size=batch_size,
                            epochs=nb_epoch,
                            verbose=0,
                            validation_data=(In_test, Out_test[:,idx:idx + n_beams]))
        
        filehandler = open('history.pkl', 'wb') 
        pickle.dump(history.history, filehandler)
        filehandler.close()
        
        AP_models.append(model)
    return AP_models



# # Training process (Normal Behaviour)
# 
# Here we will train the RF beamforming codeword prediction model with out any attacker. 




from sklearn.model_selection import train_test_split
count=count+1
DL_size=int(num_user_tot*DL_size_ratio)

np.random.seed(2016)
n_examples = DL_size
num_train  = int(DL_size * 0.8)
num_test   = int(num_user_tot*.2)

train_index = np.random.choice(range(0,num_user_tot), size=num_train, replace=False)
rem_index = set(range(0,num_user_tot))-set(train_index)
test_index= list(set(np.random.choice(list(rem_index), size=num_test, replace=False)))

In_train = In_set[train_index]
In_test =  In_set[test_index] 

Out_train = Out_set[train_index]
Out_test = Out_set[test_index]


#In_train, In_test, Out_train, Out_test =  train_test_split(In_set, Out_set, test_size=0.33)

# Learning model parameters
nb_epoch = 10   
batch_size = 128 #100  
dr = 0.05                  # dropout rate  
num_hidden_layers=4
nodes_per_layer=In_train.shape[1]
loss_fn='mean_squared_error'

eps = 2.0 * 16.0 / 255.0


# Model training
AP_models = train(In_train, Out_train, In_test, Out_test,
                                      nb_epoch, batch_size,dr,
                                      num_hidden_layers, nodes_per_layer,
                                      loss_fn,num_tot_TX,num_beams)

# Model running/testing
DL_Result={}
mse_list = []
for id in range(0,num_tot_TX,1): 

    beams_predicted=AP_models[id].predict( In_test, batch_size=10, verbose=0)

    DL_Result['TX'+str(id+1)+'Pred_Beams']=beams_predicted
    DL_Result['TX'+str(id+1)+'Opt_Beams']=Out_test[:,id*num_beams:(id+1)*num_beams]

    mse = mean_squared_error(Out_test[:,id*num_beams:(id+1)*num_beams],beams_predicted)

    mse_list.append(mse)
    mse_training=np.mean(mse_list)
print('mse:',np.mean(mse_list))

DL_Result['user_index']=test_index




import warnings
warnings.filterwarnings('ignore')
# Model running/testing
DL_Result={}
mse_list = []
for id in range(0,num_tot_TX,1): 
    # !!!!! Attack generation !!!!
    In_test_adv = fgsm(AP_models[id], In_test,Out_test,num_tot_TX,num_beams,eps)
    beams_predicted=AP_models[id].predict( In_test_adv, batch_size=128, verbose=0)

    DL_Result['TX'+str(id+1)+'Pred_Beams']=beams_predicted
    DL_Result['TX'+str(id+1)+'Opt_Beams']=Out_test[:,id*num_beams:(id+1)*num_beams]

    mse = mean_squared_error(Out_test[:,id*num_beams:(id+1)*num_beams],beams_predicted)
    mse_list.append(mse)
    mse_undef=np.mean(mse_list)
print('mse:',np.mean(mse_list))




# Model training function
def adv_train(In_train, Out_train, In_test, Out_test,
          nb_epoch, batch_size,dr,
          num_hidden_layers, nodes_per_layer,
          loss_fn,n_BS,n_beams, eps):
    
    in_shp = list(In_train.shape[1:])

    AP_models = []
    mcp_save = ModelCheckpoint('model.hdf5', save_best_only=True, verbose=0, 
                                    monitor='val_mean_squared_error', mode='min')
    
    for idx in range(0, n_BS*n_beams-2, n_beams):
        idx_str = str(idx / n_beams + 1)
        act_func = 'relu'
        model = Sequential()
        model.add(Dense(94, input_dim=in_shp[0], activation=act_func))
        model.add(Dense(94, activation=act_func))
        model.add(Dense(94, activation=act_func))
        model.add(Dense(94, activation=act_func))
        model.add(Dense(n_beams, activation=act_func))
        model.compile(loss=loss_fn, optimizer='ADAM', metrics=['mean_squared_error'])
        
        history = model.fit(In_train,
                            Out_train[:, idx:idx + n_beams],
                            batch_size=batch_size,
                            epochs=nb_epoch,
                            verbose=0,
                            validation_data=(In_test, Out_test[:,idx:idx + n_beams]))
        
        callbacks = [mcp_save]
        for _ in range(10):
            In_train_adv = fgsm(model, In_train,Out_train, n_BS, n_beams)
            In_train_adv = np.concatenate((In_train, In_train_adv), axis=0)
            
            Out_train_adv = np.concatenate((Out_train, Out_train), axis=0)
            
            history = model.fit(In_train_adv,
                                Out_train_adv[:, idx:idx + n_beams],
                                batch_size=batch_size,
                                epochs=nb_epoch*3,
                                verbose=0,
                                callbacks=callbacks,
                                validation_data=(In_test, Out_test[:,idx:idx + n_beams]))
            model.load_weights('model.hdf5')
        
        AP_models.append(model)
    return AP_models



AP_models = adv_train(In_train, Out_train, In_test, Out_test,
                      nb_epoch, batch_size,dr,
                      num_hidden_layers, nodes_per_layer,
                      loss_fn,num_tot_TX,num_beams,eps)




# Model running/testing 
DL_Result={}
mse_list = []
for id in range(0,num_tot_TX,1): 
    # !!!!! Attack generation !!!!
    In_test_adv = fgsm(AP_models[id], In_test,Out_test,num_tot_TX,num_beams,eps)
    beams_predicted=AP_models[id].predict( In_test_adv, batch_size=10, verbose=0)

    DL_Result['TX'+str(id+1)+'Pred_Beams']=beams_predicted
    DL_Result['TX'+str(id+1)+'Opt_Beams']=Out_test[:,id*num_beams:(id+1)*num_beams]

    mse = mean_squared_error(Out_test[:,id*num_beams:(id+1)*num_beams],beams_predicted)
    mse_list.append(mse)
    mse_def=np.mean(mse_list)
print('mse:',np.mean(mse_list))

output_without_op = pd.DataFrame({"mse_training":[mse_training],"mse_undef":[mse_undef],"mse_def":[mse_def]})
output_without_op.to_csv(os.path.join("output_without_op", "all.csv"), mode='a', index=False,header=False) 


#vals = [mse_training,mse_undef,0.0003265942230835527]
#df = pd.DataFrame({'Scenario':['SC1: Normal','SC2: Attacked','SC3: Defended'],'vals':vals})
#df.plot.bar(x='Scenario',y='vals', rot=0,color=['green', 'red','blue'], 
#            edgecolor='black',legend='',linewidth=2)
#plt.xlabel('')
#ax = plt.gca()
#for i, v in enumerate(vals):
#    ax.text(-0.17+i*0.97, 0.0002 + v, str(np.round(vals[i],5)), color='blue')
#plt.ylabel('MSE')
#plt.grid(True, which="both", ls="--")
#plt.show()




