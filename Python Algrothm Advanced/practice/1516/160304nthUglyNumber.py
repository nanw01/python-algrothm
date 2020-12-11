# ### Ex.4 Ugly Number II
# Write a program to find the n-th ugly number.
# Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. 
# For example, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 is the sequence of the first 10 ugly numbers.
# Note that 1 is typically treated as an ugly number.

import heapq


def nthUglyNumber(n):
    q2, q3, q5 = [2], [3], [5]
    ugly = 1
    for u in heapq.merge(q2, q3, q5):

        if n == 1:
            return ugly
        if u > ugly:
            ugly = u
            n -= 1
            q2 += 2 * u,
            q3 += 3 * u,
            q5 += 5 * u,


print(nthUglyNumber(17))
