class Solution(object):
    def addBinary(self, a, b):
        """
        input: string a, string b
        return: string
        """
        # write your solution here

        l1, l2 = len(a), len(b)
        if l1 < l2:
            l1, l2 = l2, l1
            a, b = b, a
        la, lb = [int(x) for x in a[::-1]], [int(x) for x in b[::-1]]

        for i in range(l1):
            if i < l2:
                la[i] += lb[i]

            if la[i] > 1:
                la[i] -= 2

                if i == l1 - 1:
                    la.append(1)
                else:
                    la[i + 1] += 1

        return ''.join(str(x) for x in la[::-1])


s = Solution()
a = '01'
b = '10'
print(s.addBinary(a, b))
