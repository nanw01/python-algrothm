# Given two integers a ≤ b, write a program that transforms a into b by a minimum sequence of increment (add 1) and unfolding (multiply by 2) operations.

# For example,

# 23 = ((5 * 2 + 1) * 2 + 1)

# 113 = ((((11 + 1) + 1) + 1) * 2 * 2 * 2 + 1)

def intSeq(a, b):
    if a == b:
        return str(a)

    if b % 2 == 1:
        return '('+intSeq(a, b-1)+' + 1)'

    if b < a*2:
        return '('+intSeq(a, b-1)+' + 1)'

    return intSeq(a, b/2)+' * 2'


a = 61
b = 911
print(str(b) + " = " + intSeq(a, b))
