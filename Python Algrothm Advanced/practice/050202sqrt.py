# ### Ex.2: sqrt(x)

# Implement int sqrt(int x).
# Compute and return the square root of x.
# x is guaranteed to be a non-negative integer.


def sqrt(x):
    if x == 0:
        return 0

    left, right = 1, x
    while left <= right:
        mid = left+(right-left)//2
        if(mid == x//mid):
            return mid

        if mid < x//mid:
            left = mid+1
        else:
            right = mid-1
    return right


def sqrtNewton(x):

    r = x
    while r*r > x:
        r = (r+x//r)//2
    return r


print(sqrt(40))

print(sqrtNewton(125348))
