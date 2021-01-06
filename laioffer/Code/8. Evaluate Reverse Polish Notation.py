class Solution(object):
    def evalRPN(self, tokens):
        """
        input: string[] tokens
        return: int
        """
        # write your solution here
        lst = []
        for i in tokens:
            if i.isdigit():
                lst.append(i)
            elif i == '+':
                a = lst.pop()
                b = lst.pop()
                lst.append(int(b) + int(a))
            elif i == '-':
                a = lst.pop()
                b = lst.pop()
                lst.append(int(b) - int(a))
            elif i == '*':
                a = lst.pop()
                b = lst.pop()
                lst.append(int(b) * int(a))
            elif i == '/':
                a = lst.pop()
                b = lst.pop()
                lst.append(int(b) / int(a))
            else:
                return - 1

        return lst[0]


if __name__ == "__main__":
    s = Solution()
    x = ["0", "12", "4", "+", "-"]
    print(s.evalRPN(x))
