import json
import pandas as pd
from collections import defaultdict

with open('dataset/raw_data-20190221T150829Z-001/raw_data/train_data.json') as f:
    data = json.load(f)
    print(len(data))
    df = pd.DataFrame(data)
    newDict = defaultdict(int)
    chara = ['.', '?', '\'s', ':', '!', '-', ';', ',', '\"', '\'', '(', ')', '[', ']']
    for i in range(len(data)):
        simple = data[i]
        oracle = simple['oracle']
        for j in chara:
            oracle = oracle.replace(j, '')
        ora = oracle.split(' ')
        for x in range(len(ora)):
            newDict[ora[x].lower()] += 1
    dataFrame = pd.DataFrame(pd.Series(newDict), columns=['frequency'])
    dataFrame = dataFrame.reset_index().rename(columns={'index': 'token'})
    dataFrame.to_csv('dataset/holl_input_output.oracle.vocab', sep='\t', index=None, header=None)
print('Count Complete')
