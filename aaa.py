from random import randint


# Function to convert a given number to an Excel column
def getColumnName(n):

    result = ""
    while n > 0:
        index = (n - 1) % 26
        result += chr(index + ord('A'))
        n = (n - 1) // 26
    return result[::-1]


if __name__ == '__main__':

    print(getColumnName(28))
