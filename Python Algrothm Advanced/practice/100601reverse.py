# 1.0.1  Ex.1 Reverse a String

def reverse(s):
    l = []
    for i in range(len(s)):
        l.append(s[i])

    r = []
    while len(l) != 0:
        r.append(l.pop())
    return ''.join(r)


s = "hello world"
print(reverse(s))

s = "madamimadam"
print(reverse(s))