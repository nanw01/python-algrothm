class Solution(object):
    def decodeString(self, input):
        """
        input: string input
        return: string
        """
        # write your solution here

        lst = []

        for i in input:
            if i.isdigit():
                if len(lst) == 0 or not lst[-1].isdigit():
                    lst.append(i)
                else:
                    lst.append(lst.pop() + i)

            elif i == '[':
                lst.append(i)
            elif i.isalpha():
                if len(lst) == 0 or not lst[-1].isalpha():
                    lst.append(i)
                else:
                    lst.append(lst.pop()+i)
            elif i == ']':

                cc = lst.pop()
                lst.pop()
                counts = lst.pop()

                if len(lst) == 0 or not lst[-1].isalpha():
                    lst.append(cc * int(counts))
                else:
                    lst[-1] = lst[-1]+cc*int(counts)

            else:
                pass
        return ''.join(lst)


s = Solution()
print(s.decodeString("la6[i7[o8[f]f9[e]]10[r]]"))
