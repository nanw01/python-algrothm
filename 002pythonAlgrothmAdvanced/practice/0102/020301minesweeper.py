import random


def minesweeper(m, n, p):

    # 将棋盘扩充一圈
    board = [[None] * (n+2) for i in range(m+2)]
    # add nums
    for i in range(1,m+1):
        for j in range(1,n+1):
            r=random.random()
            board[i][j] = -1 if r<p else 0

    # print board
    for i in range(1,m+1):
        for j in range(1,n+1):
            print('*',end=' ') if board[i][j]==-1 else print('.',end=' ')
        print()

    # counts
    for i in range(1,m+1):
        for j in range(1,n+1):
            if board[i][j] == 0:
                for ii in range(i-1,i+2):
                    for jj in range(j-1,j+2):
                        if board[ii][jj] == -1:
                            board[i][j]+=1


    # print counts
    for i in range(1,m+1):
        for j in range(1,n+1):
            print('*',end=' ') if board[i][j]==-1 else print(board[i][j],end=' ')
        print()
    



    


minesweeper(5, 5, 0.3)
