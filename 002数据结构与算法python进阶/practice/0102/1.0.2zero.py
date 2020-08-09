# 给一个m×n的矩阵，如果有一个元素为0，则把该元素对应的行与列所有元素全部变成0。


def zero(matrix):
    m = [None]*len(matrix)
    n = [None]*len(matrix[0])

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if(matrix[i][j]==0):
                m[i]=1
                n[j]=1
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if(m[i]==1 or n[j]==1):
                matrix[i][j]=0



matrix = [[1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


for x in matrix:
    print(x, sep=" ")

print('\n')

zero(matrix)
for x in matrix:
    print(x, sep=" ")