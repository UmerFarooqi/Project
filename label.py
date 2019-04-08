import json
import random
import numpy as np

all_data = []
with open('/Users/muhammadumerfarooqi/Documents/FYP/data_files/all_data.json', 'r') as f:
    all_data = json.load(f)

with open('/Users/muhammadumerfarooqi/Documents/FYP/data_files/all_data2.json', 'r') as f:
    all_data2 = json.load(f)

all_data = all_data + all_data2

#print(len(all_data))

NUM_TEST = 3500
NUM_TRAIN = len(all_data) - NUM_TEST
#np.random.shuffle(all_data)
#NUM_VAL = 2000
#print(np.shape(all_data))
#all_data_shuffled = random.shuffle(all_data, random)


#Acceleration, Aggression, Agility, Balance, Ball Control, Composure, Crossing, Curve, DEF, DRI,
#Dribbling, Finishing, Free Kick Accuracy, Heading Accuracy, Interceptions, Jumping, Long Passing,
#Long Shots, Marking, OVA, PAC, PAS, Penalties, PHY, Positioning, POT, Reactions, SHO, Short Passing,
#Shot Power, Sliding Tackle, Sprint Speed, Stamina, Standing Tackle, Strength, Vision, Volleys

NUM_FEATURES = 27
NUM_100SCALEFEATURES = 27
features = np.zeros((len(all_data),NUM_FEATURES))
style = np.zeros(len(all_data))

#print(len(style))
'''i = 0
for i in range(len(all_data)):
    j = 0
    features[i][j] = all_data[i]['old_details']['crossing']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['finishing']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['heading_accuracy']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['short_passing']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['volleys']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['dribbling']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['curve']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['long_passing']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['ball_control']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['acceleration']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['sprint_speed']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['agility']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['reactions']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['balance']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['shot_power']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['jumping']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['stamina']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['strength']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['long_shots']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['aggression']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['interceptions']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['positioning']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['vision']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['penalties']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['marking']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['standing_tackle']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    features[i][j] = all_data[i]['old_details']['sliding_tackle']
    if features[i][j] == 0:
        print(all_data[i]['player_id'])
    j+=1
    #features[i][j] = all_data[i]['old_details']['age']

    i+=1'''
#print(i)
i = 0
for i in range(len(all_data)):
    style[i] = all_data[i]['current_details']['rating']

#print(min(style))
#print(max(style))

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
    elif all_data[i]['current_details']['rating'] >= 80 and style[i] < 90:
        all_data_80.append(all_data[i])
    elif all_data[i]['current_details']['rating'] >= 70 and style[i] < 80:
        all_data_70.append(all_data[i])
    elif all_data[i]['current_details']['rating'] >= 60 and style[i] < 70:
        all_data_60.append(all_data[i])
    elif all_data[i]['current_details']['rating'] >= 50 and style[i] < 60:
        all_data_50.append(all_data[i])
    elif all_data[i]['current_details']['rating'] >= 40 and style[i] < 50:
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

print(len(all_data))
print(len(all_data_train))
print(len(all_data_val))
print(len(all_data_test))
'''i = 0
for i in range(len(all_data)):
    if all_data[i]['current_details']['rating'] >=60 and all_data[i]['current_details']['rating'] < 65:
        if all_data[i]['output'] == 1:
            style[i] = 0
        elif all_data[i]['output'] == 2:
            style[i] = 8
        elif all_data[i]['output'] == 3:
            style[i] = 16
        elif all_data[i]['output'] == 4:
            style[i] = 24
        elif all_data[i]['output'] == 5:
            style[i] = 32
        elif all_data[i]['output'] == 6:
            style[i] = 40
        elif all_data[i]['output'] == 7:
            style[i] = 48
    elif all_data[i]['current_details']['rating'] >=65 and all_data[i]['current_details']['rating'] < 70:
        if all_data[i]['output'] == 1:
            style[i] = 1
        elif all_data[i]['output'] == 2:
            style[i] = 9
        elif all_data[i]['output'] == 3:
            style[i] = 17
        elif all_data[i]['output'] == 4:
            style[i] = 25
        elif all_data[i]['output'] == 5:
            style[i] = 33
        elif all_data[i]['output'] == 6:
            style[i] = 41
        elif all_data[i]['output'] == 7:
            style[i] = 49
    elif all_data[i]['current_details']['rating'] >=70 and all_data[i]['current_details']['rating'] < 75:
        if all_data[i]['output'] == 1:
            style[i] = 2
        elif all_data[i]['output'] == 2:
            style[i] = 10
        elif all_data[i]['output'] == 3:
            style[i] = 18
        elif all_data[i]['output'] == 4:
            style[i] = 26
        elif all_data[i]['output'] == 5:
            style[i] = 34
        elif all_data[i]['output'] == 6:
            style[i] = 42
        elif all_data[i]['output'] == 7:
            style[i] = 50
    elif all_data[i]['current_details']['rating'] >=75 and all_data[i]['current_details']['rating'] < 80:
        if all_data[i]['output'] == 1:
            style[i] = 3
        elif all_data[i]['output'] == 2:
            style[i] = 11
        elif all_data[i]['output'] == 3:
            style[i] = 19
        elif all_data[i]['output'] == 4:
            style[i] = 27
        elif all_data[i]['output'] == 5:
            style[i] = 35
        elif all_data[i]['output'] == 6:
            style[i] = 43
        elif all_data[i]['output'] == 7:
            style[i] = 51
    elif all_data[i]['current_details']['rating'] >=80 and all_data[i]['current_details']['rating'] < 85:
        if all_data[i]['output'] == 1:
            style[i] = 4
        elif all_data[i]['output'] == 2:
            style[i] = 12
        elif all_data[i]['output'] == 3:
            style[i] = 20
        elif all_data[i]['output'] == 4:
            style[i] = 28
        elif all_data[i]['output'] == 5:
            style[i] = 36
        elif all_data[i]['output'] == 6:
            style[i] = 44
        elif all_data[i]['output'] == 7:
            style[i] = 52
    elif all_data[i]['current_details']['rating'] >=85 and all_data[i]['current_details']['rating'] < 90:
        if all_data[i]['output'] == 1:
            style[i] = 5
        elif all_data[i]['output'] == 2:
            style[i] = 13
        elif all_data[i]['output'] == 3:
            style[i] = 21
        elif all_data[i]['output'] == 4:
            style[i] = 29
        elif all_data[i]['output'] == 5:
            style[i] = 37
        elif all_data[i]['output'] == 6:
            style[i] = 45
        elif all_data[i]['output'] == 7:
            style[i] = 53
    elif all_data[i]['current_details']['rating'] >=90 and all_data[i]['current_details']['rating'] < 95:
        if all_data[i]['output'] == 1:
            style[i] = 6
        elif all_data[i]['output'] == 2:
            style[i] = 14
        elif all_data[i]['output'] == 3:
            style[i] = 22
        elif all_data[i]['output'] == 4:
            style[i] = 30
        elif all_data[i]['output'] == 5:
            style[i] = 38
        elif all_data[i]['output'] == 6:
            style[i] = 46
        elif all_data[i]['output'] == 7:
            style[i] = 54
    elif all_data[i]['current_details']['rating'] >=95 and all_data[i]['current_details']['rating'] < 100:
        if all_data[i]['output'] == 1:
            style[i] = 7
        elif all_data[i]['output'] == 2:
            style[i] = 15
        elif all_data[i]['output'] == 3:
            style[i] = 23
        elif all_data[i]['output'] == 4:
            style[i] = 31
        elif all_data[i]['output'] == 5:
            style[i] = 39
        elif all_data[i]['output'] == 6:
            style[i] = 47
        elif all_data[i]['output'] == 7:
            style[i] = 55'''

#print(all_data[0]['short_name'])
#print(all_data[0]['player_id'])
#print(features[0][0])
#print(style[0])

xtr = features[:NUM_TRAIN][:]
print(np.shape(xtr))
ytr = style[:NUM_TRAIN]
print(np.shape(ytr))

xva = features[NUM_TRAIN:NUM_TRAIN+NUM_VAL][:]
yva = prices[NUM_TRAIN:NUM_TRAIN+NUM_VAL]
xte = features[NUM_TRAIN+NUM_VAL:len(all_data)][:]
yte = style[NUM_TRAIN+NUM_VAL:len(all_data)]

#print(len(ytr))
#for i in range(len(ytr)):
#    print(ytr[i])
