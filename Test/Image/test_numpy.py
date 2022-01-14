##List

# m = [[1,2],[3,4]]
# print(m[0][0]) #1
# print(m[0][1]) #2
# print(m[1][0]) #3
# print(m[1][1]) #4

m1 = [[1,2],[3,4]]
m2 = [[5,6],[7,8]]

# for i in range(2):
#     for j in range(2):
#         print("i = ",i)
#         print("j = ",j)
#     print("increase i")

# for i in range(2):
#     for j in range(2):
#         print(m1[i][j])

# for i in range(2):
#     for j in range(2):
#         print(m2[i][j])

m = [[0,0],[0,0]]

for i in range(2):
    for j in range(2):
        m[i][j] = m1[i][j] + m2[i][j]

print(m)