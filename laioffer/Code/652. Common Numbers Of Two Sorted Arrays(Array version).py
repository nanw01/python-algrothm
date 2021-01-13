class Solution(object):
    def common(self, A, B):
        """
        input: int[] A, int[] B
        return: Integer[]
        """
        # write your solution here
        from collections import Counter

        dic1 = Counter(A)
        dic2 = Counter(B)
        res = []
        for key, val in dic1.items():
            res += [key]*min(val, dic2[key])
        return res


print(Solution().common([1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8, 9, 0]))
