def reverse(s):
    l = []
    for i in range(len(s)):
        l.append(s[i])

    r = []
    while len(l) != 0:
        r.append(l.pop())
    return ''.join(r)



def isPalindrome(s):
    r = reverse(s)
    return r == s

s = "hello world"
print(isPalindrome(s))

s = "madamimadam"
print(isPalindrome(s))