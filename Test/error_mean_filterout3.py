import pandas as pd
import numpy as np

model_name = ['VGG16+Dropout','VGG16','VGG16+Batch Normalization','Xception+Dropout','Xception','AlexNet']
mean = []
max_value = []
max_number = [] #No. of image that has max error
min_value = []
min_number = [] #No. of image that has min error

name = ['image', 'blank1', 'state', 'blank2', 'truth_x', 'truth_y', 'blank3', 'estimate_x', 'estimate_y', 'blank4', 'error']

for i in range(len(model_name)):
    df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Result Files/'+model_name[i]+'/result.csv',header = None,names = name)
    
    #Check rows that close (state = 1)
    close = []
    for i in range(len(df)):
        if df['state'][i] == 1:
            close.append(str(i))

    #Filter out state = 1(close)
    for j in range(len(close)):
        df.drop(int(close[j]), inplace = True)

    #Error
    mean.append(df['error'].mean())
    max_value.append(df['error'][df['error'].argmax()])
    max_number.append(df['error'].argmax()) #Row
    min_value.append(df['error'][df['error'].argmin()])
    min_number.append(df['error'].argmin()) #Row

error = {
    "model_name" : model_name,
    "mean" : mean,
    "max_value" : max_value,
    "max_number" : max_number,
    "min_value" : min_value,
    "min_number" : min_number
} 

dfw = pd.DataFrame(error)
dfw.to_csv("C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Code/Test/test_error_AllModel.csv")