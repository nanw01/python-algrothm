def hanoi(n, start, end, by):
    if (n == 1):
        print(start + " ---->  " + end)

    else:
        hanoi(n - 1, start, by, end)
        hanoi(1, start, end, by)
        hanoi(n - 1, by, end, start)


n = 3
hanoi(n, "x", "y", "z")