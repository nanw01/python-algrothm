class Solution(object):
    def isValid(self, input):
        """
        input: string input
        return: boolean
        """
        # write your solution here
        temp = []
        for i in input:
            if i == '[' or i == '(' or i == '{':
                temp.append(i)
            elif i == ']' or i == '}' or i == ')':
                if len(temp) == 0:
                    return False
                elif i == ']' and temp[-1] == '[':
                    temp.pop()
                elif i == ')' and temp[-1] == '(':
                    temp.pop()
                elif i == '}' and temp[-1] == '{':
                    temp.pop()
                else:
                    return False
        return len(temp) == 0


ss = Solution()
s = ""
print(ss.isValid(s))
s = "{}"
print(ss.isValid(s))
s = "{{}}"
print(ss.isValid(s))
s = "{{}}{}{}"
print(ss.isValid(s))
s = "{{{}}{}{{}}}"
print(ss.isValid(s))
s = "[]{}([{}]{})"
print(ss.isValid(s))
