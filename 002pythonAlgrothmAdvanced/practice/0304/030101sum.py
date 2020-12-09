def mysum_recursive(n):
    if n == 0:
        return 0
    return n + mysum_recursive(n-1)


result = mysum_recursive(11)
print(result)