
#  * 跳 n 极台阶的跳法
def f(n):
    if (n == 1):
        return 1
    if (n == 2):
        return 2
    return f(n-1) + f(n-2)


print(f(4))
