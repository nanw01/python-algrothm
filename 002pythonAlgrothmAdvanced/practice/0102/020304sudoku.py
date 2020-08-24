# 检测数独的完整性
def sudoku(matrix):

    n = len(matrix)
    result_row = result_col = result_blk = 0

    for i in range(n):
        result_row = result_col = result_blk = 0
        for j in range(n):
            # check row
            temp = matrix[i][j]
            if ((result_row&(1 << (temp-1)))==0):
                




    pass
