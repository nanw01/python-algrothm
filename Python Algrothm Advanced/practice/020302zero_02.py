# 给一个m×n的矩阵，如果有一个元素为0，则把该元素对应的行与列所有元素全部变成0。


def zero(matrix):
    m, n = [None]*len(matrix), [None]*len(matrix[0])

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                m[i], n[j] = 0, 0

    for i in range(len(m)):
        for j in range(len(n)):
            if m[i]==0 or n[j]==0:
                matrix[i][j] = 0




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
