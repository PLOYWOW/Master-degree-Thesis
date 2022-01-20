import cv2
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/File Path/file_path.csv')
file_path = list(df['file_path'])

model_name = ['VGG16+Dropout','VGG16','VGG16+Batch Normalization','Xception+Dropout','Xception','AlexNet']

for i in range(len(model_name)):
    j = 0
    #truth_x
    df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Extract Results/'+model_name[i]+'/truth_x_'+model_name[i]+'.csv')
    truth_x = list(df['truth_x'])
    #truth_y
    df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Extract Results/'+model_name[i]+'/truth_y_'+model_name[i]+'.csv')
    truth_y = list(df['truth_y'])
    #estimate_x
    df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Extract Results/'+model_name[i]+'/estimate_x_'+model_name[i]+'.csv')
    estimate_x = list(df['estimate_x'])
    #estimate_y
    df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Extract Results/'+model_name[i]+'/estimate_y_'+model_name[i]+'.csv')
    estimate_y = list(df['estimate_y'])
    for j in range(len(file_path)):
        #Read Image
        image = cv2.imread('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Data/D2/'+file_path[j])
        #Correct pt
        cv2.circle(image,(int(truth_x[j]),int(truth_y[j])),2,(255,255,255),-1) #BGR White
        cv2.circle(image,(int(truth_x[j]),int(truth_y[j])),1,(255,128,0),-1) #BGR Blue #Correct pt
        #Detected pt
        cv2.circle(image,(int(estimate_x[j]),int(estimate_y[j])),2,(255,255,255),-1) #BGR White
        cv2.circle(image,(int(estimate_x[j]),int(estimate_y[j])),1,(204,153,255),-1) #BGR Pink #Detected pt
        #Save Image
        cv2.imwrite('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Plotted Images/'+model_name[i]+'/'+model_name[i]+'_Plotted_Image_'+str(j)+'.jpg' ,image)
    

