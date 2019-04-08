### NEURAL NETWORK ###
# USC Sprigng 2017 EE500 Final Project - Sourya Dey
# Uses Deep Learning library Keras 1.1.1 <https://keras.io/> with backend Theano

#%% Imports and constants
import numpy as np
np.set_printoptions(threshold=np.inf) #View full arrays in console
import matplotlib.pyplot as plt
import os
import pickle
from pprint import pprint
import json
import random
from keras import utils as np_utils
from keras.layers import Dense
from keras.models import Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras import optimizers
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.models import load_model
from keras.layers import Dropout

all_data = []
#with open('/Users/muhammadumerfarooqi/Documents/FYP/data_files/all_data.json', 'r') as f:
with open('/home/muhammad/umer-env/all_data.json', 'r') as f:
    data = json.load(f)
    all_data = data
#with open('/Users/muhammadumerfarooqi/Documents/FYP/data_files/all_data2.json', 'r') as f:
with open('/home/muhammad/umer-env/all_data2.json', 'r') as f:
    all_data2 = json.load(f)

all_data = all_data + all_data2

all_data_90 = []
all_data_80 = []
all_data_70 = []
all_data_60 = []
all_data_50 = []
all_data_40 = []

i=0
for i in range(len(all_data)):
    
    if all_data[i]['current_details']['rating'] >= 90:
        all_data_90.append(all_data[i])
    elif all_data[i]['current_details']['rating'] >= 80 and all_data[i]['current_details']['rating'] < 90:
        all_data_80.append(all_data[i])
    elif all_data[i]['current_details']['rating'] >= 70 and all_data[i]['current_details']['rating'] < 80:
        all_data_70.append(all_data[i])
    elif all_data[i]['current_details']['rating'] >= 60 and all_data[i]['current_details']['rating'] < 70:
        all_data_60.append(all_data[i])
    elif all_data[i]['current_details']['rating'] >= 50 and all_data[i]['current_details']['rating'] < 60:
        all_data_50.append(all_data[i])
    elif all_data[i]['current_details']['rating'] >= 40 and all_data[i]['current_details']['rating'] < 50:
        all_data_40.append(all_data[i])


all_data_train = []
all_data_val = []
all_data_test = []
i = 0
for i in range(len(all_data_90)):
    if i <= (int(len(all_data_90))*0.75):
        all_data_train.append(all_data_90[i])
        i+=1
        j = i
    elif i <= j+(int(len(all_data_90))*0.15):
        all_data_val.append(all_data_90[i])
        i+=1
    else:
        all_data_test.append(all_data_90[i])
        i+=1

i = 0   
for i in range(len(all_data_80)):
    if i <= (int(len(all_data_80))*0.75):
        all_data_train.append(all_data_80[i])
        i+=1
        j = i
    elif i <= j+(int(len(all_data_80))*0.15):
        all_data_val.append(all_data_80[i])
        i+=1
    else:
        all_data_test.append(all_data_80[i])
        i+=1

i = 0   
for i in range(len(all_data_70)):
    if i <= (int(len(all_data_70))*0.75):
        all_data_train.append(all_data_70[i])
        i+=1
        j = i
    elif i <= j+(int(len(all_data_70))*0.15):
        all_data_val.append(all_data_70[i])
        i+=1
    else:
        all_data_test.append(all_data_70[i])
        i+=1

i = 0   
for i in range(len(all_data_60)):
    if i <= (int(len(all_data_60))*0.75):
        all_data_train.append(all_data_60[i])
        i+=1
        j = i
    elif i <= j+(int(len(all_data_60))*0.15):
        all_data_val.append(all_data_60[i])
        i+=1
    else:
        all_data_test.append(all_data_60[i])
        i+=1

i = 0   
for i in range(int(len(all_data_50))):
    if i <= (int(len(all_data_50))*0.75):
        all_data_train.append(all_data_50[i])
        i+=1
        j = i
    elif i <= j+(int(len(all_data_50))*0.15):
        all_data_val.append(all_data_50[i])
        i+=1
    else:
        all_data_test.append(all_data_50[i])
        i+=1

i = 0   
for i in range(int(len(all_data_40))):
    if i <= (int(len(all_data_40))*0.75):
        all_data_train.append(all_data_40[i])
        i+=1
        j = i
    elif i <= j+(int(len(all_data_40))*0.15):
        all_data_val.append(all_data_40[i])
        i+=1
    else:
        all_data_test.append(all_data_40[i])
        i+=1

NUM_TEST = len(all_data_test)
NUM_TRAIN = len(all_data_train)
NUM_VAL = len(all_data_val)

all_data = all_data_test + all_data_train + all_data_val
print(len(all_data))
#all_data_shuffled = random.shuffle(all_data, random)


NUM_FEATURES = 27
features = np.zeros((len(all_data),NUM_FEATURES))
style = np.zeros(len(all_data))

i = 0
for i in range(len(all_data)):
    j = 0
    features[i][j] = all_data[i]['old_details']['crossing']
    j+=1
    features[i][j] = all_data[i]['old_details']['finishing']
    j+=1
    features[i][j] = all_data[i]['old_details']['heading_accuracy']
    j+=1
    features[i][j] = all_data[i]['old_details']['short_passing']
    j+=1
    features[i][j] = all_data[i]['old_details']['volleys']
    j+=1
    features[i][j] = all_data[i]['old_details']['dribbling']
    j+=1
    features[i][j] = all_data[i]['old_details']['curve']
    j+=1
    features[i][j] = all_data[i]['old_details']['long_passing']
    j+=1
    features[i][j] = all_data[i]['old_details']['ball_control']
    j+=1
    features[i][j] = all_data[i]['old_details']['acceleration']
    j+=1
    features[i][j] = all_data[i]['old_details']['sprint_speed']
    j+=1
    features[i][j] = all_data[i]['old_details']['agility']
    j+=1
    features[i][j] = all_data[i]['old_details']['reactions']
    j+=1
    features[i][j] = all_data[i]['old_details']['balance']
    j+=1
    features[i][j] = all_data[i]['old_details']['shot_power']
    j+=1
    features[i][j] = all_data[i]['old_details']['jumping']
    j+=1
    features[i][j] = all_data[i]['old_details']['stamina']
    j+=1
    features[i][j] = all_data[i]['old_details']['strength']
    j+=1
    features[i][j] = all_data[i]['old_details']['long_shots']
    j+=1
    features[i][j] = all_data[i]['old_details']['aggression']
    j+=1
    features[i][j] = all_data[i]['old_details']['interceptions']
    j+=1
    features[i][j] = all_data[i]['old_details']['positioning']
    j+=1
    features[i][j] = all_data[i]['old_details']['vision']
    j+=1
    features[i][j] = all_data[i]['old_details']['penalties']
    j+=1
    features[i][j] = all_data[i]['old_details']['marking']
    j+=1
    features[i][j] = all_data[i]['old_details']['standing_tackle']
    j+=1
    features[i][j] = all_data[i]['old_details']['sliding_tackle']
    #j+=1
    #features[i][j] = all_data[i]['old_details']['age']
    i+=1
#print(i)
i = 0
for i in range(len(all_data)):
    style[i] = all_data[i]['current_details']['rating']

xtr = features[:NUM_TRAIN][:]
ytr = style[:NUM_TRAIN]
xva = features[NUM_TRAIN:NUM_TRAIN+NUM_VAL][:]
yva = style[NUM_TRAIN:NUM_TRAIN+NUM_VAL]
xte = features[NUM_TRAIN+NUM_VAL:len(all_data)][:]
yte = style[NUM_TRAIN+NUM_VAL:len(all_data)]

ytr.astype(int)
yva.astype(int)
yte.astype(int)

'''c = list(zip(xtr, ytr))
random.shuffle(c)
xtr, ytr = zip(*c)

c = list(zip(xva, yva))
random.shuffle(c)
xva, yva = zip(*c)

c = list(zip(xte, yte))
random.shuffle(c)
xte, yte = zip(*c)'''
#classes = int(max(style)-min(style)+1)
classes = 100

ytr = np_utils.to_categorical(ytr, num_classes=classes, dtype='int32')
yva = np_utils.to_categorical(yva, num_classes=classes, dtype='int32')
yte = np_utils.to_categorical(yte, num_classes=classes, dtype='int32')

#%% Data preprocessing
def normalize(features):
    ''' Normalize features by converting to mean=0, std=1) '''
    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)
    features = (features-mu)/sigma
    return features

'''def shuffle_data(features,prices,rounded_prices):
     #Shuffle features 
    np.random.seed(0) #To maintain consistency across runs
    perm = np.random.permutation(md.NUM_TOTAL)
    temp_features = np.zeros_like(features)
    temp_prices = np.zeros_like(prices)
    temp_rounded_prices = np.zeros_like(rounded_prices)
    for p in xrange(len(perm)):
        temp_features[p][:] = features[perm[p]][:]
        temp_prices[p][:] = prices[perm[p]][:]
        temp_rounded_prices[p] = rounded_prices[perm[p]]
    return (temp_features,temp_prices,temp_rounded_prices)'''

features = normalize(features)

nin = len(xtr[1])
nout = classes

#%% Neural network
def genmodel(num_units, actfn='relu', reg_coeff=0.0, last_act='softmax'):
    '''
    Source: Nitin Kamra, for USC Fall 2016 CSCI567 HW4
    Modified and used by Sourya Dey with permission
    Generate a neural network model of approporiate architecture
    Glorot normal initialization used for all layers
        num_units: architecture of network in the format [n1, n2, ... , nL]
        actfn: activation function for hidden layers ('relu'/'sigmoid'/'linear'/'softmax')
        reg_coeff: L2-regularization coefficient
        last_act: activation function for final layer ('relu'/'sigmoid'/'linear'/'softmax')
    Returns model: Keras sequential model with appropriate fully-connected architecture
    '''
    model = Sequential()
    for i in range(1, len(num_units)):
        if i == 1 and i < len(num_units) - 1: #Input layer
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=actfn, 
                            W_regularizer=Reg.l2(l=reg_coeff), kernel_initializer='glorot_normal'))
        elif i == 1 and i == len(num_units) - 1: #Input layer, network has only 1 layer
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=last_act, 
                            W_regularizer=Reg.l2(l=reg_coeff), kernel_initializer='glorot_normal'))
        elif i < len(num_units) - 1: #Hidden layer
            model.add(Dense(output_dim=num_units[i], activation=actfn, 
                            W_regularizer=Reg.l2(l=reg_coeff), kernel_initializer='glorot_normal'))
            #model.add(Dropout(0.8))
        elif i == len(num_units) - 1: #Output layer
            model.add(Dense(output_dim=num_units[i], activation=last_act, 
                            W_regularizer=Reg.l2(l=reg_coeff), kernel_initializer='glorot_normal'))
    return model


#2000, 1500, 500
#[512, 256, 128, 64]
#len(xtr)
def testmodels(xtr,ytr,xte,yte,xva,yva,num_epoch=10000, batch_size=200, actfn='relu', last_act='softmax',
               EStop=False, verbose=1, archs=[[64, 64, 32]], reg_coeffs=[5e-4],
               sgd_lrs=0.1, sgd_decays=0.001, sgd_moms=0.8, sgd_Nesterov=True,
               results_file='results2.txt'):
    '''
    Source: Nitin Kamra, for USC Fall 2016 CSCI567 HW4
    Modified and used by Sourya Dey with permission
    Train and test neural network architectures with varying parameters
        xtr, ytr, xte, yte: (Training and test) (features and prices)
        archs: List of architectures. ONLY ENTER hidden layer sizes
        actfn: activation function for hidden layers ('relu'/'sigmoid'/'linear'/'softmax')
        last_act: activation function for final layer ('relu'/'sigmoid'/'linear'/'softmax')
        reg_coeffs: List of L2-regularization coefficients
        num_epoch: number of iterations for SGD
        batch_size: batch size for gradient descent
        sgd_lr: Learning rate for SGD
        sgd_decays: List of decay parameters for the learning rate
        sgd_moms: List of momentum coefficients, works only if sgd_Nesterov = True
        sgd_Nesterov: Boolean variable to use/not use momentum
        EStop: Boolean variable to use/not use early stopping
        verbose: 0 or 1 to determine whether keras gives out training and test progress report
    Returns the Keras model with best accuracy
    Commented out: Keras model with least mean square error
    '''
    f = open(os.path.dirname(os.path.realpath(__file__))+'/result_files/'+results_file,'w')
    best_acc = 0
    best_config = []
    best_model = None
#    best_mse = np.inf
#    best_config_mse = []
#    best_model_mse = None
    call_ES = EarlyStopping(monitor='val_acc', patience=50, verbose=1, mode='auto')

    for arch in archs:
        arch.insert(0,nin)
        arch.append(nout)
        for reg_coeff in reg_coeffs:
            #for sgd_lr in sgd_lrs:
                #for sgd_decay in sgd_decays:
                    #for sgd_mom in sgd_moms:
            #print ('Starting architecture = {0}, lambda = {1}, eta = {2}, decay = {3}, momentum = {4}, actfn = {5}'.format(arch, reg_coeff, sgd_lr, sgd_decay, sgd_mom, actfn))
            #print ('Starting architecture = {0}, lambda = {1}, eta = {2}, decay = {3}, momentum = {4}, actfn = {5}'.format(arch, reg_coeff, actfn))
            model = genmodel(num_units=arch, actfn=actfn, reg_coeff=reg_coeff, last_act=last_act)
            #sgd = SGD(lr=sgd_lrs, decay=sgd_decays, momentum=sgd_moms, nesterov=sgd_Nesterov)
            sgd = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
            #sgd = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            # Train Model
            if EStop:
                model.fit(xtr,ytr, epochs=num_epoch, batch_size=batch_size, verbose=verbose, 
                          callbacks=[call_ES], validation_data=(xva,yva))
            else:
                model.fit(xtr,ytr, epochs=num_epoch, batch_size=batch_size, verbose=verbose, validation_data=(xva,yva), shuffle= True)
            # Evaluate Models
            #results = model.predict_classes(xte, verbose = 1)
            score = model.evaluate(xte,yte, batch_size=batch_size, verbose=verbose)
            if score[1] > best_acc:
                best_acc = score[1]
                #best_config = [arch, reg_coeff, sgd_lr, sgd_decay, sgd_mom, actfn, best_acc]
                best_config = [arch, reg_coeff, actfn, best_acc]
                best_model = model
            '''results = model.predict_classes(xte, verbose = 1)
            for i in range(len(yte)):
                print(yte[i])
                print("fuck")
                print(results[i])'''
#                        if score[2] < best_mse:
#                            best_mse = score[2]
#                            best_config_mse = [arch, reg_coeff, sgd_lr, sgd_decay, sgd_mom, actfn, best_mse]
#                            best_model_mse = model
            result = 'Score for architecture = {0}, lambda = {1} actfn = {2}: Acc = {3}%\n'.format(arch, reg_coeff, actfn, score[1]*100)
            print (result)
            f.write(result)
    final_result_acc = 'Best Config: architecture = {0}, lambda = {1}, actfn = {2}, best_acc = {3}%\n'.format(best_config[0], best_config[1], best_config[2], best_config[3]*100)
    print (final_result_acc)
    f.write(final_result_acc)
#    final_result_mse = 'Best Config MSE: architecture = {0}, lambda = {1}, eta = {2}, decay = {3}, momentum = {4}, actfn = {5}, best_mse = {6}\n'.format(best_config_mse[0], best_config_mse[1], best_config_mse[2], best_config_mse[3], best_config_mse[4], best_config_mse[5], best_config_mse[6])
#    print final_result_mse
#    f.write(final_result_mse)
    f.close()
    return best_model


def neighbor_accuracy(model,xte,yte, neighbor_range=2, num=NUM_TEST):
    ''' Returns percentage of correct = (predicted label is +/- n from accurate label)
        Eg: If n=2, it's essentially top5 because predicted label can be accurate label -2, -1, +0, +1, +2
        For some reason this fails if num=1, i.e. single cases can't be tested '''
    y_pred = model.predict_classes(xte[:num],verbose=0)
    y_true = np.argmax(yte[:num],axis=1)
    acc = [np.abs(y_pred[i]-y_true[i])<=neighbor_range for i in range(num)]
    return 100.0*acc.count(True)/num

def price_error(model,xte,rounded_prices,num=NUM_TEST):
    ''' Returns absolute error between predicted price and actual price, percentage absolute error, and their averages
        Pass the entire rounded_prices into this, it will automatically extract what's required
    '''
    rounded_prices = rounded_prices[NUM_TRAIN:NUM_TRAIN+num]
    y_pred = model.predict_classes(xte[:num],verbose=0)
    pred_prices = [prices_bins[i] for i in y_pred]
    error = [np.abs(pred_prices[i]-rounded_prices[i]) for i in range(num)]
    pc_error = [100.0*error[i]/rounded_prices[i] for i in range(num)]
    avg_error = np.mean(error)
    avg_pc_error = np.mean(pc_error)
    return (error,pc_error,avg_error,avg_pc_error)


########## MAIN EXECUTION ##########
#%% Trial
'''model = testmodels(xtr,ytr,xte,yte, num_epoch=2,
                             archs=[[300]],
                             results_file = 'trial.txt')

#%% Vary batch sizes only
model,model_mse = testmodels(xtr,ytr,xte,yte, batch_size=1, 
                             archs=[[nin,1000,nout]],
                             results_file = 'batch_size.txt')

#%% Vary architectures only
archs = [[nin,a,nout] for a in xrange(100,5001,100)]
model,model_mse = testmodels(xtr,ytr,xte,yte, 
                             archs=archs,
                             results_file = 'archs_1hiddenlayer.txt')

#%% Vary activation functions over architectures
#Possiblities are [relu,soft], [sigm,soft], [tanh,soft], [relu,sigm], [sigm,sigm], [tanh,sigm]
#[relu,soft] is default. Here I'm trying the next 3
archs = [[nin,a,nout] for a in xrange(500,4501,1000)]
model = testmodels(xtr,ytr,xte,yte, actfn='sigmoid',
                   archs=archs,
                   results_file = 'act_sigmoid.txt')
model = testmodels(xtr,ytr,xte,yte, actfn='tanh',
                   archs=archs,
                   results_file = 'act_tanh.txt')
model = testmodels(xtr,ytr,xte,yte, last_act='sigmoid',
                   archs=archs,
                   results_file = 'lastact_sigmoid.txt')

#%% Vary number of hidden layers in architecture only
archs = [[nin,2000,3000,nout],[nin,2000,2000,nout],[nin,2000,1500,nout],[nin,2000,1000,nout],[nin,2000,500,nout],[nin,2000,nout,nout]]
model = testmodels(xtr,ytr,xte,yte,
                   archs=archs,
                   results_file = 'archs_2hiddenlayers.txt')
model = testmodels(xtr,ytr,xte,yte,
                   archs=[[nin,2200,1500,nout]],
                   results_file = 'particular_2hiddenlayer.txt')

#%% Vary eta and archs
sgd_lrs = [1e-6,1e-5,5e-5,5e-4,1e-3,1e-2]
archs = [[nin,900,nout],[nin,1700,nout],[nin,2100,nout],[nin,2900,nout],[nin,3900,nout],[nin,4600,nout]]
model = testmodels(xtr,ytr,xte,yte,
                   archs=archs,
                   sgd_lrs=sgd_lrs,
                   results_file = 'etas_1hiddenlayer.txt')
model = testmodels(xtr,ytr,xte,yte,
                   archs=[[nin,3900,1500,nout]],
                   sgd_lrs=sgd_lrs,
                   results_file = 'etas_particular_2hiddenlayer.txt')

#%% Vary eta and archs (>1 hidden layer)
model = testmodels(xtr,ytr,xte,yte,
                   archs=[[2100]],
                   sgd_lrs=np.arange(0.02,0.1,0.01),
                   results_file = 'dump.txt')
archs = [[2000,500],[2000,1000],[2000,1500],[2000,2000],[2000,2500],[2000,1500,500],[2000,1500,1000],[2000,1500,1500],[2000,1500,2000],[2000,1500,2500]]
sgd_lrs = [0.008,0.01,0.033,0.067,0.1,0.133,0.167,0.2,0.25,0.3,0.35,0.4]
model = testmodels(xtr,ytr,xte,yte,
                   archs=archs,
                   sgd_lrs=sgd_lrs,
                   results_file = 'etas_archs_manyhiddenlayers.txt')
model = testmodels(xtr,ytr,xte,yte,
                   results_file = 'reg_1pm3_lessbatchsize.txt')'''

#%% Final
model = testmodels(xtr,ytr,xte,yte,xva,yva)
top3acc = neighbor_accuracy(model,xte,yte, neighbor_range=1)
print(top3acc)
top5acc = neighbor_accuracy(model,xte,yte)
print(top5acc)
           
#%% Post-processing
model.save(os.path.dirname(os.path.realpath(__file__))+'/model_files/final_model2.h5')
#del model
'''model = load_model(os.path.dirname(os.path.realpath(__file__))+'/model_files/final_epoch100_batch20.h5')
top3acc = neighbor_accuracy(model,xte,yte, neighbor_range=1)
top5acc = neighbor_accuracy(model,xte,yte)
error,pc_error,avg_error,avg_pc_error = price_error(model,xte,rounded_prices)'''

#%% Save variables, if desired

store_file = open('final_storevar2.txt','bw')
pickle.dump((xte,yte,top3acc,top5acc),store_file)
store_file.close()