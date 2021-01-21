class Solution(object):
    def common(self, A, B):
        """
        input: int[] A, int[] B
        return: Integer[]
        """
        # write your solution here
        from collections import Counter
        count_A = Counter(A)
        count_B = Counter(B)
        res = []

        for key, val in count_A.items():
            print(key, val)
            res += [key]*min(val, count_B[key])

        return sorted(res)


print(Solution().common([1, 2, 3, 2], [3, 4, 2, 2, 2]))
