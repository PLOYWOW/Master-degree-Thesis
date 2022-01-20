import pandas as pd

model_name = ['VGG16+Dropout','VGG16','VGG16+Batch Normalization','Xception+Dropout','Xception','AlexNet']

n = ['image', 'blank1', 'state', 'blank2', 'truth_x', 'truth_y', 'blank3', 'estimate_x', 'estimate_y', 'blank4', 'error']
for i in range(len(model_name)):
    df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Result Files/'+model_name[i]+'/result.csv',header = None,names=n)
    truth_x = list(df["truth_x"])
    truth_y = list(df["truth_y"])
    estimate_x = list(df["estimate_x"])
    estimate_y = list(df["estimate_y"])
    truth_x = pd.DataFrame(truth_x, columns = ['truth_x'])
    truth_y = pd.DataFrame(truth_y, columns = ['truth_y'])
    estimate_x = pd.DataFrame(estimate_x, columns = ['estimate_x'])
    estimate_y = pd.DataFrame(estimate_y, columns = ['estimate_y'])

    truth_x.to_csv("C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Extract Results/"+model_name[i]+"/"+"truth_x_"+model_name[i]+".csv")
    truth_y.to_csv("C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Extract Results/"+model_name[i]+"/"+"truth_y_"+model_name[i]+".csv")
    estimate_x.to_csv("C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Extract Results/"+model_name[i]+"/"+"estimate_x_"+model_name[i]+".csv")
    estimate_y.to_csv("C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Extract Results/"+model_name[i]+"/"+"estimate_y_"+model_name[i]+".csv")