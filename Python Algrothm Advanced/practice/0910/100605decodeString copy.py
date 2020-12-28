
def decodeString(s):
    stack = []
    stack.append(['', 1])
    num = ''
    for c in s:
        if c.isdigit():
            num += c
        elif c == '[':
            stack.append(['', int(num)])
            num=''
        elif c == ']':
            c, k = stack.pop()
            stack[-1][0] += c*k 
        else:
            stack[-1][0]+=c

    return stack[0][0]


s = "3[a]2[bc]"
print(decodeString(s))
s = "3[a1[c]]"
print(decodeString(s))
s = "2[abc]3[cd]ef"
print(decodeString(s))
s = "12[ab]"
print(decodeString(s))
