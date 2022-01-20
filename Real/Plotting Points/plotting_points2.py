import cv2
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('file_path.csv')
file_path = list(df['file_path'])

df = pd.read_csv('truth_x.csv')
truth_x = list(df['truth_x'])

df = pd.read_csv('truth_y.csv')
truth_y = list(df['truth_y'])

df = pd.read_csv('estimate_x.csv')
estimate_x = list(df['estimate_x'])

df = pd.read_csv('estimate_y.csv')
estimate_y = list(df['estimate_y'])

for i in range(len(file_path)):
    #Read Image
    image = cv2.imread('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Code/Real/Plotting Points/'+file_path[i])
    #Correct pt
    cv2.circle(image,(int(truth_x[i]),int(truth_y[i])),2,(255,255,255),-1) #BGR White
    cv2.circle(image,(int(truth_x[i]),int(truth_y[i])),1,(255,128,0),-1) #BGR Blue #Correct pt
    #Detected pt
    cv2.circle(image,(int(estimate_x[i]),int(estimate_y[i])),2,(255,255,255),-1) #BGR White
    cv2.circle(image,(int(estimate_x[i]),int(estimate_y[i])),1,(204,153,255),-1) #BGR Pink #Detected pt
    #Save Image
    cv2.imwrite('C:/Users/ployw/OneDrive/Desktop/Kyutech/Lab/Code/Real/Plotting Points/Plotted Images/'+'Plotted_Image_'+str(i)+'.jpg' ,image)

