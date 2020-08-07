# Given a non-negative integer represented as a non-empty array of digits, plus one to the integer.
# You may assume the integer do not contain any leading zero, except the number 0 itself.
# The digits are stored such that the most significant digit is at the head of the list.


def plusOne(digits):
    if len(digits) == 0:
        return False

    addCarrty = 1
    for i in range(len(digits)-1,-1,-1):
        digits[i] += addCarrty
        if digits[i] == 10:
            digits[i] = 0
            if i == 0:
                digits.insert(0,1)
        else:
            break
    
    return digits

digits = [1, 2, 3]
print(plusOne(digits))
digits = [9, 9, 9]
print(plusOne(digits))