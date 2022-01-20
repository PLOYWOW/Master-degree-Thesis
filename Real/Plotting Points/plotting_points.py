import cv2
import matplotlib.pyplot as plt

A = cv2.imread('kan1_2017052212_24_6.png')
B = cv2.imread('kan1_2017052212_46_18.png')
C = cv2.imread('kan1_2017052212_11_6.png')
D = cv2.imread('kan1_2017052212_11_10.png')
print(type(A))
print(A.shape)
print(A.dtype)
print(A.min())
print(A.max())

#A
#Correct pt
cv2.circle(A,(53,44),2,(255,255,255),-1) #BGR White
cv2.circle(A,(53,44),1,(255,128,0),-1) #BGR Blue #Correct pt

#Detected pt
cv2.circle(A,(int(52.7537879943847),int(43.7013206481933)),2,(255,255,255),-1) #BGR White
cv2.circle(A,(int(52.7537879943847),int(43.7013206481933)),1,(204,153,255),-1) #BGR Pink #Detected pt

# plt.imshow(A[:,:,::-1])
# plt.show()

cv2.imwrite('A.png',A)

#B
#Correct pt
cv2.circle(B,(49,37),2,(255,255,255),-1) #BGR White
cv2.circle(B,(49,37),1,(255,128,0),-1) #BGR Blue #Correct pt

#Detected pt
cv2.circle(B,(int(48.529369354248),int(36.5374717712402)),2,(255,255,255),-1) #BGR White
cv2.circle(B,(int(48.529369354248),int(36.5374717712402)),1,(204,153,255),-1) #BGR Pink #Detected pt

# plt.imshow(B[:,:,::-1])
# plt.show()

cv2.imwrite('B.png',B)

#C
#Correct pt
cv2.circle(C,(28,48),2,(255,255,255),-1) #BGR White
cv2.circle(C,(28,48),1,(255,128,0),-1) #BGR Blue #Correct pt

#Detected pt
cv2.circle(C,(int(58.0826606750488),int(41.0015335083007)),2,(255,255,255),-1) #BGR White
cv2.circle(C,(int(58.0826606750488),int(41.0015335083007)),1,(204,153,255),-1) #BGR Pink #Detected pt

# plt.imshow(C[:,:,::-1])
# plt.show()

cv2.imwrite('C.png',C)

#D
#Correct pt
cv2.circle(D,(28,50),2,(255,255,255),-1) #BGR White
cv2.circle(D,(28,50),1,(255,128,0),-1) #BGR Blue #Correct pt

#Detected pt
cv2.circle(D,(int(60.4882888793945),int(42.8990859985351)),2,(255,255,255),-1) #BGR White
cv2.circle(D,(int(60.4882888793945),int(42.8990859985351)),1,(204,153,255),-1) #BGR Pink #Detected pt

# plt.imshow(D[:,:,::-1])
# plt.show()

cv2.imwrite('D.png',D)
