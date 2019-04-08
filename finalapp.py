#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup as bs
import requests
import re
import json
from random import randint
offset = 0
all_ids = []
all_versions = [ # Versions of Fifa (year and subversion) on the website
    {'v': '12', 'e': '156820'},
    {'v': '13', 'e': '157396'},
    {'v': '14', 'e': '157760'},
    {'v': '15', 'e': '158116'},
    {'v': '16', 'e': '158494'},
    {'v': '17', 'e': '158857'},
    {'v': '18', 'e': '159214'}
]
try:
    with open('ProgressTracker1.json') as f: #change to test
        all_ids = json.load(f)
except:
    print("Starting fresh...")

# In[2]: Gets the html code of the url provided in the argument


def soup_maker(url):
    r = requests.get(url)
    markup = r.content
    soup = bs(markup, 'lxml')
    return soup


# In[85]: Finds the list of top players in the list


def find_top_players(soup, v, e):
    all_data = []
    table = soup.find('table', {'class': 'table'})
    tbody = table.find('tbody')
    all_a = tbody.find_all('a', {'href': re.compile("^/player/")})
    for idx, player in enumerate(all_a):
        try:
            final_details = {}
            player_id = int(re.search('([0-9])\w+', player['href']).group(0))
            if player_id not in all_ids:
                final_details['short_name'] = player.text
                final_details['player_id'] = player_id
                #print(str(idx+offset) + ': ' + player.text)
                old_details = player_all_details('http://sofifa.com' + player['href'] + '?v='+v+'&e='+e+'&set=true')
                final_details.update({'old_details': old_details})
                current_details = player_all_details('https://sofifa.com/'+ player['href'] + '/')
                final_details.update({'current_details': current_details})
                #print(final_details)
                all_data.append(final_details)
                #print(player_id)
                all_ids.append(player_id)
                #print(final_details['current_details']['short_passing'])
                try:
                    defending = int((final_details['current_details']['marking']+final_details['current_details']['standing_tackle']+final_details['current_details']['sliding_tackle'])/3)
                    
                    if final_details['current_details']['short_passing'] >= 88:
                        final_details['output'] = 1
                    if final_details['current_details']['potential'] >= 80:
                        if final_details['current_details']['short_passing'] >= 85 and ("M" in final_details['current_details']['position'] or "W" in final_details['current_details']['position']) and defending > 70:
                            final_details['output'] = 1
                        elif ("M" in final_details['current_details']['position'] or "W" in final_details['current_details']['position']) and defending > 65 and final_details['current_details']['finishing'] >=75:
                            final_details['output'] = 3
                        elif ("M" in final_details['current_details']['position'] or "W" in final_details['current_details']['position']) and defending > 65 and final_details['current_details']['aggression'] >=70:
                            final_details['output'] = 2
                        elif ("M" in final_details['current_details']['position'] or "W" in final_details['current_details']['position']) and final_details['current_details']['finishing'] >=75:
                            final_details['output'] = 4
                        elif ("M" in final_details['current_details']['position'] or "W" in final_details['current_details']['position']):
                            final_details['output'] = randint(2, 4)

                        if final_details['current_details']['short_passing'] >= 80 and ("F" in final_details['current_details']['position'] or "T" in final_details['current_details']['position']) and defending > 65:
                            final_details['output'] = 1
                        elif ("F" in final_details['current_details']['position'] or "T" in final_details['current_details']['position']):
                            final_details['output'] = randint(2, 4)

                        if final_details['current_details']['short_passing'] >= 79 and (final_details['current_details']['position'][-1:] == "B"):
                            final_details['output'] = 1
                        elif (final_details['current_details']['position'][-1:] == "B"):
                            final_details['output'] = randint(2, 4)

                    elif final_details['current_details']['potential'] >= 70:
                        if (final_details['current_details']['position'][-1:] == "M" or final_details['current_details']['position'][-1:] == "W") and final_details['current_details']['short_passing'] >= 75:
                            final_details['output'] = 5
                        elif (final_details['current_details']['position'][-1:] == "M" or final_details['current_details']['position'][-1:] == "W") and final_details['current_details']['short_passing'] <= 65 and final_details['current_details']['strength'] >=80:
                            final_details['output'] = 7
                        elif (final_details['current_details']['position'][-1:] == "M" or final_details['current_details']['position'][-1:] == "W") and defending >= 75:
                            final_details['output'] = 6
                        elif (final_details['current_details']['position'][-1:] == "M" or final_details['current_details']['position'][-1:] == "W"):
                            final_details['output'] = randint(5, 6)

                        if final_details['current_details']['short_passing'] >= 72 and (final_details['current_details']['position'][-1:] == "F" or final_details['current_details']['position'][-1:] == "T"):
                            final_details['output'] = 5
                        elif (final_details['current_details']['position'][-1:] == "F" or final_details['current_details']['position'][-1:] == "T") and defending >= 68 and final_details['current_details']['short_passing'] >= 65:
                            final_details['output'] = 6
                        elif (final_details['current_details']['position'][-1:] == "F" or final_details['current_details']['position'][-1:] == "T") and final_details['current_details']['short_passing'] < 65 and final_details['current_details']['strength'] >=80:
                            final_details['output'] = 7
                        elif (final_details['current_details']['position'][-1:] == "F" or final_details['current_details']['position'][-1:] == "T"):
                            final_details['output'] = randint(5, 6)

                        if final_details['current_details']['short_passing'] >= 70 and (final_details['current_details']['position'][-1:] == "B"):
                            final_details['output'] = 5
                        elif (final_details['current_details']['position'][-1:] == "B") and defending >= 78 and final_details['current_details']['short_passing'] >= 62:
                            final_details['output'] = 6
                        elif (final_details['current_details']['position'][-1:] == "B") and final_details['current_details']['short_passing'] < 62 and final_details['current_details']['strength'] >=80:
                            final_details['output'] = 7
                        elif (final_details['current_details']['position'][-1:] == "F" or final_details['current_details']['position'][-1:] == "T"):
                            final_details['output'] = randint(6, 7)

                    elif final_details['current_details']['potential'] >= 60:
                        if (final_details['current_details']['position'][-1:] == "M" or final_details['current_details']['position'][-1:] == "W") and final_details['current_details']['short_passing'] >= 70:
                            final_details['output'] = 5
                        elif (final_details['current_details']['position'][-1:] == "M" or final_details['current_details']['position'][-1:] == "W") and final_details['current_details']['short_passing'] <= 62 and final_details['current_details']['strength'] >=80:
                            final_details['output'] = 7
                        elif (final_details['current_details']['position'][-1:] == "M" or final_details['current_details']['position'][-1:] == "W") and defending >= 68:
                            final_details['output'] = 6
                        elif (final_details['current_details']['position'][-1:] == "M" or final_details['current_details']['position'][-1:] == "W"):
                            final_details['output'] = randint(6, 7)

                        if final_details['current_details']['short_passing'] >= 68 and (final_details['current_details']['position'][-1:] == "F" or final_details['current_details']['position'][-1:] == "T"):
                            final_details['output'] = 5
                        elif (final_details['current_details']['position'][-1:] == "F" or final_details['current_details']['position'][-1:] == "T") and defending >= 65 and final_details['current_details']['short_passing'] >= 62:
                            final_details['output'] = 6
                        elif (final_details['current_details']['position'][-1:] == "F" or final_details['current_details']['position'][-1:] == "T") and final_details['current_details']['short_passing'] < 62 and final_details['current_details']['strength'] >=80:
                            final_details['output'] = 7
                        elif (final_details['current_details']['position'][-1:] == "F" or final_details['current_details']['position'][-1:] == "T"):
                            final_details['output'] = randint(5, 6)

                        if final_details['current_details']['short_passing'] >= 68 and (final_details['current_details']['position'][-1:] == "B"):
                            final_details['output'] = 5
                        elif (final_details['current_details']['position'][-1:] == "B") and defending >= 72 and final_details['current_details']['short_passing'] >= 60:
                            final_details['output'] = 6
                        elif (final_details['current_details']['position'][-1:] == "B") and final_details['current_details']['short_passing'] < 60 and final_details['current_details']['strength'] >=80:
                            final_details['output'] = 7
                        elif (final_details['current_details']['position'][-1:] == "F" or final_details['current_details']['position'][-1:] == "T"):
                            final_details['output'] = randint(6, 7)

                    else: final_details['output'] = 0
                    
                except: 
                    final_details['output'] = 0
                    #print('fuck off')

                print(player.text+ ": " + str(final_details['output']) + " " + str(final_details['current_details']['rating']))

            else:
                print(str(idx+offset) + ': ' + player.text + ' (Ignored)')
        except KeyboardInterrupt:
            exit()
        except:
            print(str(idx+offset) + ': ' + player.text + ' (Error)') # If any info is missing, then the data is not recorded
            final_details['output'] = 0
    return all_data


# In[22]: Find basic info of the Player (Name, Dob, Age)


def find_player_info(soup):
    player_data = {}
    span = soup.find('div', attrs={'class': 'meta'}) # .text.strip()
    player_data['full_name'] = str(span)[str(span).find('>')+1: str(span).find('<a href=')].strip()
    dob = re.search('(\(.{5,6}, .{4}\))', str(span)).group(0)
    player_data['dob'] = dob.replace('(', '').replace(')', '')
    #infos = span.replace(dob + ' ', '').split(' ') #infos = span.replace(dob + ‘ ‘, ‘’).split(‘ ‘)
    span = span.text.strip()
    player_data['age'] = int(span[span.index(('Age'))+4: span.index(('Age'))+6])
    #player_data['position'] = span[span.index(('Age'))-2: span.index(('Age'))-0]
    str1 = str(span)[str(span).find(')')+4: str(span).find(')')+7].strip()
    if str1.isalpha():
        player_data['position'] = str(span)[str(span).find(')')+4: str(span).find(')')+7].strip()
    else: player_data['position'] = span[span.index(('Age'))-2: span.index(('Age'))-0]
    #print(player_data['position'])
    '''try:
        if len(soup.find_all('ul')) > 2:
            player_data['country'] = soup.find_all('ul')[2].find('a').text
            #print(player_data['country'])
    except:
        player_data['country'] = "Unknown"
        #print(player_data['country'])'''
    return(player_data)


# In[36]: Find the overall ratings of the player


def find_player_stats(soup):
    player_data = {}
    info = re.findall('\d+', soup.text)
    player_data['rating'] = int(info[0])
    player_data['potential'] = int(info[1])
#     player_data['value'] = int(info[2])
#     player_data['wage'] = int(info[3])
    return(player_data)



# In[80]: Find all the stats of the player (eg. acceleration, fk_accuracy, etc)
fifa_stats = ['Crossing', 'Finishing', 'Heading Accuracy',
 'Short Passing', 'Volleys', 'Dribbling', 'Curve',
 'Free Kick Accuracy', 'Long Passing', 'Ball Control',
 'Acceleration', 'Sprint Speed', 'Agility', 'Reactions',
 'Balance', 'Shot Power', 'Jumping', 'Stamina', 'Strength',
 'Long Shots', 'Aggression', 'Interceptions', 'Positioning',
 'Vision', 'Penalties', 'Composure', 'Marking', 'Standing Tackle',
 'Sliding Tackle']

def find_fifa_info(soup):
    player_data = {}
    '''all_lists = soup.find_all('ul', {'class': 'pl'})
    for i, one_list in enumerate(all_lists):
        if i == 0:
            player_data['preff_foot'] = one_list.find('label', text='Preferred Foot').parent.contents[2].strip('\n ')
            player_data['weak_foot'] = int(one_list.find('label', text='Weak Foot').parent.contents[2].strip('\n '))
            player_data['skill_moves'] = int(one_list.find('label', text='Skill Moves').parent.contents[2].strip('\n '))
        elif i!=1 and i!=2 and i!=9 and i!=10:
            all_stats = one_list.find_all('li')
            #print (all_stats)
            for stat in all_stats:
                stat_txt = stat.text
                attribute_name = re.search('([A-Za-z\s]+)', stat_txt).group(0).strip().lower().replace(' ', '_')
                #print(attribute_name)
                attribute_rating = int(re.search('(^[0-9]{1,3})', stat_txt).group(0).strip())
                player_data[attribute_name] = attribute_rating'''
    #divs_without_skill = soup[1].find_all(‘div’, {‘class’: ‘col-3’})[:3]
    #more_lis = [div.find_all(‘li’) for div in divs_without_skill]
    lis = soup.find_all('li')
    #print(lis)
    for li in lis:
        for stats in fifa_stats:
            if stats in li.text:
                player_data[stats.replace(' ', '_').lower()] = int(re.search('(^[0-9]{1,3})', li.text).group(0).strip())            
    return(player_data)


# In[89]: Run all the above functions and merge data collected from all of them for a single player


def player_all_details(url):
    all_details = {}
    soup = soup_maker(url)
    player_info = soup.find('div', {'class': 'player'})
    all_details.update(find_player_info(player_info))
    player_stats = soup.find('div', {'class': 'stats'})
    all_details.update(find_player_stats(player_stats))
    all_details.update(find_fifa_info(soup))
    return(all_details)


# In[84]: Running web scraping
for version in all_versions:
    v = version['v']
    e = version['e']
    offset = 0
    print("Running for Fifa " + v)
    i = 1
    while offset < 3000:
        url = 'https://sofifa.com/players?ael=22&aeh=23&v=' + v + '&e=' + e + '&set=true&offset=' + str(offset)
        url += "&pn1%5B%5D=27&pn1%5B%5D=25&pn1%5B%5D=23&pn1%5B%5D=22&pn1%5B%5D=21&pn1%5B%5D=20&pn1%5B%5D=18&pn1%5B%5D=16&pn1%5B%5D=14&pn1%5B%5D=12&pn1%5B%5D=10&pn1%5B%5D=8&pn1%5B%5D=7&pn1%5B%5D=5&pn1%5B%5D=3&pn1%5B%5D=2"
        soup = soup_maker(url)
        version_data = find_top_players(soup, v, e)
        offset += 61
        with open('Fifa_v2_' + v + '_' + str(i) + '.json', 'w') as outfile:
            json.dump(version_data, outfile)
        i += 1
        with open('ProgressTracker1.json', 'w') as outfile: # Delete this file
            json.dump(all_ids, outfile)
    print("Completed Fifa " + v)


# In[33]: Debug (Every thing below this was used to debug)


# neySoup = soup_maker('https://sofifa.com/player/190871/neymar-da-silva-santos-jr/?v=12&e=156820&set=true')
# hazSoup = soup_maker('https://sofifa.com/player/183277/eden-hazard/?v=12&e=156820&set=true')
#

# In[82]:


# print(find_fifa_info(neySoup))


# In[81]:


# print(find_fifa_info(hazSoup))


# In[ ]:




