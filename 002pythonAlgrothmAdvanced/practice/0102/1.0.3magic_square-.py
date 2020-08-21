# 打印 N*N 的幻方，各行各列对角线之和都相等

# 有固定模式，

def magic_square(n):
    magic = [[0]*(n) for i in range(n)]
    row=n-1
    col=n//2
  
    magic[row][col] = 1

    for i in range(2, n*n+1):
        try_row=(row+1)%n
        try_col=(col+1)%n

        if (magic[try_row][try_col]==0):
            row=try_row
            col=try_col
        else:
            row=(row-1+n)%n
        magic[row][col]=i

    for x in magic:
        print(x,sep=' ')

magic_square(5)