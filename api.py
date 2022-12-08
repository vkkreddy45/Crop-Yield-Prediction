import os
import pandas as pd

def apiEnd():
    df=pd.read_csv("Key_Value_Data/key_value_data.csv",names=['Country','crops'])
    tl = []
    for i in df['crops']:
        t = eval(i)
        tl.append(t)
    t2= []
    for i in df['Country']:
        t2.append(i)
        
    final_dict = dict()
    for k,v in zip(t2,tl):
        final_dict[k] = v    
    return final_dict


