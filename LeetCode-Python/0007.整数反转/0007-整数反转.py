class Solution:
    def reverse(self, x: int) -> int:
        op = 1
        if x < 0:
            op = -1
            s = str(x)[1:]
        else:
            s = str(x)
        res = op * int(s[::-1])

        return res if -2**31 <= res <= 2**31-1 else 0
