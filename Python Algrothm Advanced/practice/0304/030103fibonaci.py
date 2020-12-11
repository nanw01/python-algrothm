# fibonaci squence is the series of numbers:
# 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55,...


def fibonacci1(n):
    assert(n>=0)
    a, b = 0, 1
    for i in range(1, n+1):
        a, b = b, a + b
    return a    
    
def fibonacci2(n):
    assert(n>=0)
    if (n <= 2): 
        return 1
    return fibonacci2(n-1) + fibonacci2(n-2)

# 递归 
def fibonacci3(n):
    assert(n>=0)
    if (n <= 1): 
        return (n,0)
    (a, b) = fibonacci3(n-1)
    return (a+b, a)


def fibonacci4(n):
    assert(n>=0)
    if n<=1:
        return (n,0)
    (a,b) = fibonacci4(n-1)
    return (a+b,a)


print(fibonacci4(10))
