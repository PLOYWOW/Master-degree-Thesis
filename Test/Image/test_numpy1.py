##numpy

import numpy as np

aray1 = np.array(range(3,7))
print(aray1)
print(aray1.shape) #(4,) รูปร่างของอาเรย์
print(aray1.size) #4 จำนวนสมาชิกในอาเรย์
print(aray1.ndim) #1 จำนวนมิติของอาเรย์

aray2 = np.array([[1,2],[3,4]])
print(aray2)
print(aray2.shape) #(2, 2) รูปร่างของอาเรย์
print(aray2.size) #4 จำนวนสมาชิกในอาเรย์
print(aray2.ndim) #2 จำนวนมิติของอาเรย์

aray3 = np.array([[1,2],[3,4],[5,6]])
print(aray3)
print(aray3.shape) #(3, 2) รูปร่างของอาเรย์
print(aray3.size) #6 จำนวนสมาชิกในอาเรย์
print(aray3.ndim) #2 จำนวนมิติของอาเรย์

aray4 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(aray4)
print(aray4.shape) #(2, 2, 2) รูปร่างของอาเรย์
print(aray4.size) #8 จำนวนสมาชิกในอาเรย์
print(aray4.ndim) #3 จำนวนมิติของอาเรย์

aray5 = np.array([[1,2,3],[4,5,6]])
#1 2 3
#4 5 6
print(aray5[0][1]) # ได้ 2
print(aray5[1][2]) # ได้ 6

print(aray5[0,2]) # ได้ 3
print(aray5[1,1]) # ได้ 5

print(aray5[1][:]) # ได้ [4 5 6]
print(aray5[1,:]) # ได้ [4 5 6]

print(aray5[:,1]) # ได้ [2 5]

print(aray5[:][1]) # ได้ [4 5 6]]

aray6 = np.array([13,14,15,16,17,18,19])
print(aray6[:3]) # ได้ [13 14 15] ไม่เอา3
print(aray6[3:5]) # ได้ [16 17] เอา3 ไม่เอา5
print(aray6[5:]) # ได้ [18 19] เอา5

print(aray6[-1]) # ได้ 19
print(aray6[-3:-1]) # ได้ [17 18] เอา-3 ไม่เอา-1
print(aray6[5:-1]) # ได้ [18] เอา5 ไม่เอา-1

print(aray6[1:4:2]) # ได้ [14 16] เอา1 ไม่เอา4
print(aray6[1::2]) # ได้ [14 16 18] เอา1 
print(aray6[:6:2]) # ได้ [13 15 17] เอา ไม่เอา6
print(aray6[::2]) # ได้ [13 15 17 19] 
print(aray6[::-1]) # ได้ [19 18 17 16 15 14 13]

aray7 = np.array([[13,14,15,16],[17,18,19,20],[21,22,23,24]])
#13 14 15 16
#17 18 19 20
#21 22 23 24
print(aray7[1:2,2:3]) #ได้ [[19]]
print(aray7[0:2,1:3])# ได้
# [[14 15]
# [18 19]]
print(aray7[0,1:3]) # ได้ [14 15]
print(aray7[::2,2]) # ได้ [15 23] ข้าม2
print(aray7[::-1,::-1]) # ได้ กลับหลัง
# [[24 23 22 21]
# [20 19 18 17]
# [16 15 14 13]]
