from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import numpy as np
import os

all_data = []
with open('/Users/muhammadumerfarooqi/Documents/FYP/data_files/all_data.json', 'r') as f:
#with open('/home/muhammad/umer-env/all_data.json', 'r') as f:
    data = json.load(f)
    all_data = data
with open('/Users/muhammadumerfarooqi/Documents/FYP/data_files/all_data2.json', 'r') as f:
#with open('/home/muhammad/umer-env/all_data2.json', 'r') as f:
    all_data2 = json.load(f)

all_data = all_data + all_data2

NUM_TEST = 3500
NUM_TRAIN = len(all_data) - NUM_TEST 
#NUM_VAL = 2000
np.random.shuffle(all_data)
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
ytr.astype(int)

xte = features[NUM_TRAIN:len(all_data)][:]
yte = style[NUM_TRAIN:len(all_data)]
yte.astype(int)

model = LogisticRegression(solver='lbfgs', multi_class='auto', verbose=1, max_iter=10000)
model.fit(xtr, ytr)
predicted_classes = model.predict(xte)
accuracy = accuracy_score(yte.flatten(),predicted_classes)
parameters = model.coef_