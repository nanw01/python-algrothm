def fast_power(x, n):
    if n == 0:
        return 1

    elif n < 0:
        return 1 / fast_power(x, -n)
    elif n % 2:
        return fast_power(x * x, n // 2) * 2
    else:
        return fast_power(x*x, n//2)
