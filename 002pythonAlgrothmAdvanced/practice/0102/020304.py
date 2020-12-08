def sudoku( matrix):
    # row
    n = len(matrix)


    for i in range(n):
        result_row = result_col = result_block
        for j in range(n):
            tmp = matrix[i][j]
            if ((result_row & (1 << (tmp-1))) == 0):
                result_row = result_row | (1<<(tmp-1))
            else:
                print("row: ", i, j)
                return False





if __name__ == "__main__":
    matrix = [
    [5,3,4,6,7,8,9,1,2],
    [6,7,2,1,9,5,3,4,8],
    [1,9,8,3,4,2,5,6,7],
    [8,5,9,7,6,1,4,2,3],
    [4,2,6,8,5,3,7,9,1],
    [7,1,3,9,2,4,8,5,6],
    [9,6,1,5,3,7,2,8,4],
    [2,8,7,4,1,9,6,3,5],
    [3,4,5,2,8,6,1,7,9]]



    print(sudoku(matrix))