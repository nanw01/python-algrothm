def letterCount(s):
    freq = {}

    for c in s:
        # print(c)
        k = ''
        if c.isalpha():
            k=k.join(c) 
        if k:
            freq[k] = 1+ freq.get(k,0)

    print(freq)

    max_count = 0
    max_character =''

    for (k,v) in freq.items():
        if v > max_count:
            max_count = v
            max_character = k

    print('%s: %7d' % (max_character, max_count))        

    

s = "fh;sdhaf;sdh;fhsd;lncoiuL:JKHDL:HF:KSDFLUJSD{PIO"
letterCount(s)



from collections import Counter
def letterCount2(s):
    c = Counter(x for x in s if x != " ")

    for letter, count in c.most_common(4):
        print('%s: %7d' % (letter, count))

s = "Hello World How are you I am fine thank you and you"
letterCount2(s)