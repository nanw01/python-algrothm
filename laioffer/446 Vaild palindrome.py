def valid(input):
    """
    input: string input
    return: boolean
    """
    # write your solution here
    left, right = 0, len(input)-1
    while left < right:
        if not ((input[left].isdigit()) or (input[left] >= 'A' and input[left] <= 'Z') or (input[left] >= 'a' and input[left] <= 'z')):
            left += 1
        elif not((input[right].isdigit()) or (input[right] >= 'A' and input[right] <= 'Z') or (input[right] >= 'a' and input[right] <= 'z')):
            right -= 1
        else:
            if input[left].lower() != input[right].lower():
                return False
            else:
                left += 1
                right -= 1
    return True


l = "FQJKY$jB3Qd-fSOUAI`^iAUOSfdQ3BjyKJQF"
print(valid(l))
