
# 当插入右括号时，匹配左括号，然后判断是否

def isValid(s):
    stack = []
    for c in s:
        if (c == '(' or c == '[' or c == '{'):
            stack.append(c)
        else:
            if len(stack)==0:
                return False
            if (   (c == ')' and stack[-1] == '(')
                or (c == ']' and stack[-1] == '[')
                or (c == '}' and stack[-1] == '{')):
                stack.pop()
            else:
                return False
    return len(stack)==0


s = ""
print(isValid(s))
s = "{}"
print(isValid(s))
s = "{{}}"
print(isValid(s))
s = "{{}}{}{}"
print(isValid(s))
s = "{{{}}{}{{}}}"
print(isValid(s))
s = "[]{}([{}]{})"
print(isValid(s))



s = "{"
print(isValid(s))
s = "}"
print(isValid(s))
s = "{}}"
print(isValid(s))
s = "{}{}}"
print(isValid(s))
s = "}{"
print(isValid(s))
s = "}{}"
print(isValid(s))
s = "(]"
print(isValid(s))
s = "[}]"
print(isValid(s))