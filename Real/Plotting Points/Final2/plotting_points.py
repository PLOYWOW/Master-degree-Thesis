import cv2
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Extract File Path and Coordinate/extract_filepath_coordinate.csv')
df.rename(columns={"Unnamed: 0": "model_name", "Unnamed: 1": "number"},inplace=True)

model_name = ['VGG16+Dropout','VGG16','VGG16+Batch Normalization','Xception+Dropout','Xception','AlexNet']
model_dict = ["VGG16_Dropout","VGG16","VGG16_BatchNormalization","Xception_Dropout","Xception","AlexNet"]

for i in range(len(model_dict)):
    j = 0
    dfw = pd.DataFrame(df[df["model_name"] == model_dict[i]])
    file_path = list(dfw['file_path'])
    truth_x = list(dfw['truth_x'])
    truth_y = list(dfw['truth_y'])
    estimate_x = list(dfw['estimate_x'])
    estimate_y = list(dfw['estimate_y'])
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
        cv2.imwrite('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Thesis/Result/Pupil detection/D2 Data set/Plotted Images2/'+model_name[i]+'/'+model_name[i]+'_Plotted_Image_'+str(j)+'.jpg' ,image)
