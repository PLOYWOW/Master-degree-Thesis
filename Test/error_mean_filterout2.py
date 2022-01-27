import pandas as pd
import numpy as np

name = ['image', 'blank1', 'state', 'blank2', 'truth_x', 'truth_y', 'blank3', 'estimate_x', 'estimate_y', 'blank4', 'error']
df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/AlexNet/result.csv',header = None, names = name)

#Check rows that close (state = 1)
close = []
for i in range(len(df)):
    if df["state"][i] == 1:
        close.append(str(i))

#Print mean error before filter out
print(df["error"].mean())

#Filter out state = 1(close)
for j in range(len(close)):
    df.drop(int(close[j]), inplace = True)

#Print mean error after filter out
print(df["error"].mean())

mean = {
    "model_name" : ["AlexNet"],
    "mean_error" : [df["error"].mean()]
}

df = pd.DataFrame(mean)
df.to_csv("C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Code/Test/test_error_AlexNet.csv")
