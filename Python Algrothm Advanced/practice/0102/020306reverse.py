# 翻转字符串

# 直接使用现有方法
def reverse(s):
    return s[::-1]

s = "hello"
r = reverse(s) # O(n)
print(r)



# 新建数组遍交换位置，时间复杂度O(n/2)
def reverse2(s):
    l = list(s)
    for i in range(len(l)//2):
        l[i], l[len(s)-1-i] = l[len(s)-1-i], l[i]
    return ''.join(l)


s = "hello"
r = reverse2(s)
print(r)


# 使用while交换位置
def reverse3(s):
    l = list(s)
    begin, end = 0, len(l) - 1
    while begin <= end:
        l[begin], l[end] = l[end], l[begin]
    return ''.join(l)

s = "hello"
r = reverse3(s)
print(r)