from collections import Counter
def wordCount(s):
    wordcount = Counter(s.split())
    print(wordcount)

s = "Hello World How are you I am fine thank you and you"
wordCount(s)