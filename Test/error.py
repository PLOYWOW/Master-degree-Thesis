import pandas as pd
import numpy as np

model_name = ['VGG16+Dropout','VGG16','VGG16+Batch Normalization','Xception+Dropout','Xception','AlexNet']
mean = []
max_value = []
max_number = [] #No. of image that has max error
min_value = []
min_number = [] #No. of image that has min error

n = ['image', 'blank1', 'state', 'blank2', 'truth_x', 'truth_y', 'blank3', 'estimate_x', 'estimate_y', 'blank4', 'error']

for i in range(len(model_name)):
    df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Result Files/'+model_name[i]+'/result.csv',header = None,names=n)
    mean.append(df['error'].mean())
    max_value.append(df['error'][df['error'].argmax()])
    max_number.append(df['error'].argmax())
    min_value.append(df['error'][df['error'].argmin()])
    min_number.append(df['error'].argmin())

model_name = pd.Series(model_name,name='model_name')
mean = pd.Series(mean,name='mean')
max_value = pd.Series(max_value,name='max_value')
max_number = pd.Series(max_number,name='max_number')
min_value = pd.Series(min_value,name='min_value')
min_number = pd.Series(min_number,name='min_number')

error = pd.concat([model_name,mean,max_value,max_number,min_value,min_number],axis=1)
error.to_csv("C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Extract Results/error_all_models/error.csv")

print('a')
print('b')