class Solution(object):
    def fibonacci(self, K):
        """
        input: int K
        return: long
        """
        # write your solution here
        if K < 0:
            return -1
        if K == 0 or K == 1:
            return K
        if K == 2:
            return 1

        a = self.fibonacci(K - 2)
        b = self.fibonacci(K-1)
        return a + b

    def fibonacci2(self, K):

        a, b = 0, 1
        for _ in range(K):
            a, b = a + b, a

        return a


s = Solution()
print(s.fibonacci(2))
print(s.fibonacci2(20))
