# ### Ex.1 Word Pattern
# Given a pattern and a string str, find if str follows the same pattern.
# Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in str.
# Examples:
# pattern = "abba", str = "dog cat cat dog" should return true.
# pattern = "abba", str = "dog cat cat fish" should return false.
# pattern = "aaaa", str = "dog cat cat dog" should return false.
# pattern = "abba", str = "dog dog dog dog" should return false.


def wordPattern(pattern, str):
    s = pattern
    t = str.split()
    return len(set(zip(s, t))) == len(set(s)) == len(set(t)) and len(s) == len(t)


pattern = 'abba'
str = "dog dog dog dog"

print(wordPattern(pattern, str))


pattern = 'abba'
str = "dog cat cat dog"

print(wordPattern(pattern, str))


pattern = 'abba'
str = "dog cat dog cat"

print(wordPattern(pattern, str))