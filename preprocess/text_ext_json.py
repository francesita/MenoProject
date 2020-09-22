import pandas as pd
import string
import pickle
import json
import os
import re

'''
The following code is used to create json file of articles found online using beautiful soup. This is to have json files with a uniform format so I can use later on in preprocessing etc.
author: Frances
date: 2020-07-01
'''

# Extract data from json files
bodyText = []
titles = []

dic = {}




def make_guardian_json(json_file_list):
    '''
    Function to extract the bodytext and titles of articles in json file and store it into lists bodyText and titles
    '''
    guardianArr = []
    for json_file in json_file_list:
        for i in range(len(json_file['response']['results'])):
            if len(json_file['response']['results'][i]['tags']) != 0:
                author = json_file['response']['results'][i]['tags'][0]['webTitle']
            else:
                author = 'None'
            guardianArr.append({
                "url": json_file['response']['results'][i]['id'],
                "category":json_file['response']['results'][i]['sectionName'],
                "date": json_file['response']['results'][i]['webPublicationDate'],
                "title": json_file['response']['results'][i]['webTitle'],
                "author": author,
                "bodyText": json_file['response']['results'][i]['fields']['bodyText']
            })

            bodyText.append(json_file['response']['results'][i]['fields']['bodyText'])
        
    with open('guardian_articles.json', 'w', encoding='utf-8') as outfile:
        json.dump(guardianArr, outfile, indent=2, ensure_ascii=False)

    #return bodyText

def ext_text(json_file):
    '''
    Function to extract the bodytext and articles in json file and store into list bodyText
    '''
    artText = []
    for i in range(len(json_file)):
        bodyText.append(json_file[i]['bodyText'])
        #titles.append(json_file['response']['results'][i]['webTitle'])
    #return artText

list_json = []
def directory_ext(directory):
    '''
    iterate through directory and call ext_text function to extract corpus data
    '''

    for filename in os.listdir(directory):
        #filename = os.fsdecode(file)
        if filename.endswith('.json'):
            print(filename)
            json_f = open(os.path.join(directory, filename), "r")
            json_data = json.load(json_f)
            #ext_text(json_data)
            #bug if more than one file
            #cooment out if not in guardian
            list_json.append(json_data)
        else:
            continue
    #comment out if not from guardian
    len(list_json)
    return make_guardian_json(list_json)
    

#calling directory_Ext func
directory = '/media/frances/Expansion Drive/Menopause/code/data/guardian_articles/menopause_articles/peri_post_meno'
#directory_ext calls function to extract json text which returns a list
articles = directory_ext(directory)



'''
Saving textual data into pickle files for later use
'''

file_body = open("bodyText.pkl", "wb")

#file_dic =  open("decades.pkl", "wb")

pickle.dump(bodyText, file_body)

