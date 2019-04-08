import json


all_data = []
for v in range(12, 19):
    for i in range(1,51):
        with open('Fifa_v2_' + str(v) + '_' + str(i) + '.json') as f:
            data = json.load(f)
            all_data += data
with open('all_data2.json', 'w') as outfile:
    json.dump(all_data, outfile)
