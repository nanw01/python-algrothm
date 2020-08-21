def uglyNumber(num):
    for p in 2, 3, 5:
        while num % p == 0 < num:
            num /= p
    return num == 1


print(uglyNumber(17))