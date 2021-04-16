class Solution(object):
    def search(self, matrix, target):
        """
        input: int[][] matrix, int target
        return: int[]
        """
        # write your solution here
        a, b = -1, -1
        if len(matrix) == 0:
            return [a, b]
        if len(matrix[0]) == 0:
            return [a, b]

        left, right = 0, len(matrix)-1

        while left + 1 < right:
            mid = (left+right)//2

            if matrix[mid][-1] > target:
                right = mid

            elif matrix[mid][0] < target:
                left = mid

        if matrix[left][0] <= target <= matrix[left][-1]:
            a = left
        if matrix[right][0] <= target <= matrix[right][-1]:
            a = right

        if a == -1:
            return [a, b]

        sub_left, sub_right = 0, len(matrix[a])-1

        while sub_left+1 < sub_right:
            sub_mid = (sub_left+sub_right)//2
            if matrix[a][sub_mid] < target:
                sub_left = sub_mid
            elif matrix[a][sub_mid] >= target:
                sub_right = sub_mid

        if matrix[a][sub_left] == target:
            b = sub_left

        if matrix[a][sub_right] == target:
            b = sub_right

        if b == -1:
            return [-1, -1]
        return [a, b]


s = Solution()

matrix = [[1, 2, 3], [4, 5, 7], [8, 9, 10]]
target = 7

print(s.search(matrix, target))
