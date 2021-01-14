# ### Ex.3 First Unique Character in a String

# Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.

# Examples:

# s = "givenastring"

# return 2.

# s = "ifitdoesnotexist",

# return 1.

# Note: You may assume the string contain only lowercase letters.


def firstUniqChar(s):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    index = [s.index(l) for l in letters if s.count(l) == 1]
    return min(index) if len(index) > 0 else -1


s = "givenastring"
print(firstUniqChar(s))
s = "ifitdoesnotexist"
print(firstUniqChar(s))
s = "abcdabcd"
print(firstUniqChar(s))
