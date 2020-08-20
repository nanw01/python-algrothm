# ### Ex.10 Keyboard Row

# Given a List of words, return the words that can be typed using letters of alphabet on only one row's of American keyboard like the image below.

# <img src="../images/ch11/keyboard.png" width="560"/>

# Example 1:

# Input: ["Hello", "Alaska", "Dad", "Peace"]

# Output: ["Alaska", "Dad"]

# Note:

# You may use one character in the keyboard more than once.

# You may assume the input string will only contain letters of alphabet.


def findWords(words):
    line1, line2, line3 = set('qwertyuiop'), set('asdfghjkl'), set('zxcvbnm')
    ret = []
    for word in words:
        w = set(word.lower())
        if w.issubset(line1) or w.issubset(line2) or w.issubset(line3):
            ret.append(word)
    return ret


words = ["Hello", "Alaska", "Dad", "Peace"]
print(findWords(words))