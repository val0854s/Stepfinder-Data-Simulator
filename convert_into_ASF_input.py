import pandas as pd


#convert experimental data into input required for AutoStepfinder

D421C_mPAINT = "Y:/users/Valerie/data/Experimental_data/D421C_mPAINT.csv"
life_K560C = "Y:/users/Valerie/data/Experimental_data/life_K560C_new.csv"
life_K560N = "Y:/users/Valerie/data/Experimental_data/life_K560N.csv"


def convert_data(experimental_data):
# Read the data from the CSV file
    data = pd.read_csv(experimental_data, delimiter=',', header=None)
    ASF_data = data[1]
    #print(ASF_data)
    ASF_data.to_csv('ASF_life_K560C_new.txt', sep=' ', index=False, header=None)

convert_data(life_K560C)