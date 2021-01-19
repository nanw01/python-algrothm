class Solution(object):
    def search(self, matrix, target):
        """
        input: int[][] matrix, int target
        return: int[]
        """
        # write your solution here

        a, b = -1, -1
        for i in range(len(matrix)):
            if len(matrix[i]) > 0 and matrix[i][0] <= target <= matrix[i][-1]:
                a = i
                for j in range(len(matrix[i])):
                    if matrix[i][j] == target:
                        b = j

        if b == -1:
            a = -1
        return [a, b]


s = Solution()

matrix = [[1, 2, 3], [4, 5, 7], [8, 9, 10]]
target = 7

print(s.search(matrix, target))
