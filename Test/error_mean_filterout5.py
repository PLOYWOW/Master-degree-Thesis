import pandas as pd
import numpy as np

model_name = ['VGG16+Dropout','VGG16','VGG16+Batch Normalization','Xception+Dropout','Xception','AlexNet']
mean = []
max_number = [] #No. of image that has max error, refer with path file
max_value = []
min_number = [] #No. of image that has min error, refer with path file
min_value = []

name = ['image', 'blank1', 'state', 'blank2', 'truth_x', 'truth_y', 'blank3', 'estimate_x', 'estimate_y', 'blank4', 'error']

for i in range(len(model_name)):
    df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Result Files/'+model_name[i]+'/result.csv',header = None,names = name)
    
    #Filter out state = 1(close)
    for j in range(len(df)):
        if df['state'][j] == 1:
            df.drop(j, inplace = True)
    df = df.reset_index() #Reset inde

    #Error
    mean.append(df['error'].mean())
    max_index = df['error'].argmax() #Row, new index
    max_number.append(df['index'][max_index]) #Number of image
    max_value.append(df['error'][df['error'].argmax()])
    min_index = df['error'].argmin() #Row, new index
    min_number.append(df['index'][min_index]) #Number of image
    min_value.append(df['error'][df['error'].argmin()])

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

print(0)
