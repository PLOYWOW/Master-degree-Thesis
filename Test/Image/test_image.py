import cv2
import matplotlib.pyplot as plt

i = str(0)
hey = cv2.imread('C:/Users/ployw/OneDrive/Desktop/Others/'+i+'.png')
print(type(hey))
print(hey.shape)
print(hey.dtype)
print(hey.min())
print(hey.max())

#Detected pt
cv2.circle(hey,(20,50),3,(255,255,255),-1) #BGR White
cv2.circle(hey,(20,50),2,(204,153,255),-1) #BGR Pink #Detected pt

#Correct pt
cv2.circle(hey,(50,50),3,(255,255,255),-1) #BGR White
cv2.circle(hey,(50,50),2,(255,128,0),-1) #BGR Blue #Correct pt
plt.imshow(hey[:,:,::-1])
plt.show()

cv2.imwrite('test2.jpg',hey)



