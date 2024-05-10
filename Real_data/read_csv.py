import pandas as pd
import numpy as np

dataframe = pd.read_csv(r"/home/sebastiano/HMM_matlab/Real_data/load_file/data.csv")
pp = dataframe.iloc[1].smrbci
print(pp)

static_output = list()
for line in dataframe.smrbci:
    pp_1 = float(line[1:line.index(',')])
    pp_2 = float(line[line.index(',')+1:len(line)-1])
    vect = [pp_1,pp_2]
    
    static_output.append(vect)

print(static_output[1][0])