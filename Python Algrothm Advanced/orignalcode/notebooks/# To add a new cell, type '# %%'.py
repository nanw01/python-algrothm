# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# 乘法口诀
from random import random
import random


def mults():

    for i in range(9):
        for j in range(9):
            if i < j:
                break
            print(f'{i+1} * {j+1} = {(i+1)*(j+1)}', end='  ')
        print()


mults()


# %%
# 洗牌


def shuffle(cards):
    for i in range(len(cards)):
        j = i + random.randint(0, len(cards)-i-1)
        cards[i], cards[j] = cards[j], cards[i]
    return cards


shuffle([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])


# %%
# 数质数
def count_prime(n):
    prime_arary = [True]*(n+1)
    i = 2
    while i*i <= n:
        if prime_arary[i]:
            j = i
            while(i*j <= n):
                prime_arary[i*j] = False
                j = j+1
        i += 1
    count = 0
    for i in range(2, n+1):
        if prime_arary[i]:
            count += 1
            print(i, end=' ')
    return count


count_prime(100)


# %%
# EX1: 扫雷游戏
def minesweeper(m, n, p):
    # 创建棋盘
    board = [[None]*(n+2) for i in range(m+2)]

    # 生成点
    for i in range(1, 1+m):
        for j in range(1, 1+n):
            r = random()
            board[i][j] = -1 if r < p else 0

    # 显示器盘
    for i in range(1, 1+m):
        for j in range(1, 1+n):
            print('*', end=' ') if board[i][j] == -1 else print('.', end=' ')
        print()
    print()

    # 计算有几个雷
    for i in range(1, 1+m):
        for j in range(1, 1+n):
            if board[i][j] != -1:
                for ii in range(i-1, i+2):
                    for jj in range(j-1, j+2):
                        if board[ii][jj] == -1:
                            board[i][j] += 1
    # for i in range(1,1+m):
    #     for j in range(1,1+n):
    #         print(board[i][j],end=' ')
    #     print()
    # print()

    # 打印棋盘
    for i in range(1, 1+m):
        for j in range(1, 1+n):
            print(board[i][j], end=' ') if board[i][j] != - \
                1 else print('*', end=' ')
        print()


minesweeper(5, 10, 0.5)


# %%
# 矩阵变换
def zero(matrix):
    m = [None]*len(matrix)
    n = [None]*len(matrix[0])
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                m[i], n[j] = 1, 1

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if m[i] == 1 or n[j] == 1:
                matrix[i][j] = 0


matrix = [[1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
for x in matrix:
    print(x, sep=" ")
print()
zero(matrix)
for x in matrix:
    print(x, sep=" ")


# %%

# 验证数独
def sudoku(matrix):
    # 验证行
    res_rows = res_col = res_block = 0
    for i in range(len(matrix)):
        if sum(matrix[i]) == sum([1:9]):

            # 验证列

            # 验证块


matrix = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9]
]

sudoku(matrix)


# %%
# Ex5:旋转数组
def rotate(matrix):
    n = len(matrix)
    result = [[0]*n for i in range(n)]

    for i in range(n):
        for j in range(n):
            result[j][n-1-i] = matrix[i][j]

    for x in result:
        print(x, sep=' ')


matrix = [[i*5+j for j in range(5)] for i in range(5)]
matrix


# %%
rotate(matrix)


# %%
# Ex6: 反转字符串
def reverse(s):
    return s[::-1]


s = "hello"
r = reverse(s)  # O(n)
r


# %%
def reverse2(s):
    lst = list(s)
    for i in range(len(lst)//2):
        lst[i], lst[len(s)-1-i] = lst[len(s)-1-i], lst[i]

    return ''.join(lst)


s = "hello"
r = reverse2(s)
r


# %%
# Ex7:最长子字符串
def find_consecutive_ones(nums):
    local = maximum = 0
    for i in nums:
        local = local + 1 if i == 1 else 0
        maximum = max(local, maximum)
    return maximum


nums = [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
result = find_consecutive_ones(nums)
result


# %%
# 最大数
def largest_twice(nums):
    maxmum = second = idx = 0
    for i in range(len(nums)):
        if maxmum < nums[i]:
            second = maxmum
            maxmum = nums[i]
            idx = i
        elif second < nums[i]:
            second = nums[i]
    return idx if (maxmum >= second * 2) else -1


nums = [1, 2, 3, 8, 3, 2, 1]
result = largest_twice(nums)
result


# %%
# Ex9 find all nums disappeared in an array

# 慢
def findDisappearedNumbers1(nums):
    result = []
    for i in range(1, len(nums)+1):
        if i not in nums:
            result.append(i)
    return result

# 快


def findDisappearedNumbers2(nums):
    for i in range(len(nums)):
        index = abs(nums[i])-1
        nums[index] = -abs(nums[index])
    return[i+1 for i in range(len(nums)) if nums[i] > 0]


nums = [4, 3, 2, 7, 8, 2, 3, 1]
print(findDisappearedNumbers1(nums))
print(findDisappearedNumbers2(nums))


# %%
# Ex10 plus one
def plusOne(digits):
    if len(digits) == 0:
        return -1
    addCarry = 1
    for i in range(len(digits)-1, -1, -1):
        digits[i] += addCarry
        if digits[i] == 10:
            digits[i] = 0
            if i == 0:
                digits.insert(0, 1)
        else:
            break
    return digits


digits = [1, 2, 9]
print(plusOne(digits))
digits = [9, 9, 9]
print(plusOne(digits))


# %%
# Recursion
def mysum_recursive(n):
    if n == 0:
        return 0
    return n+mysum_recursive(n-1)


result = mysum_recursive(10)
result


# %%
def factorial(n):
    if n == 1 or n == 0:
        return 1
    return n * factorial(n-1)


factorial(5)


# %%
def fibonacci(n):
    assert n > 0
    if n <= 1:
        return (n, 0)
    (a, b) = fibonacci(n-1)
    return (a+b, a)


fibonacci(40)[0]


# %%
# Ex5 数学表达式
def intSeq(a, b):

    if a == b:
        return str(a)

    if b % 2 == 1:
        return '('+intSeq(a, b-1)+'+1)'

    if b < a*2:
        return '('+intSeq(a, b-1)+'+1)'

    return intSeq(a, b/2)+'*2'


a = 5
b = 101
print(str(b) + " = " + intSeq(a, b))


# %%
# Ex6 汉诺塔

def hanoi(n, start, end, by):

    # %%
    # subset
    # 所有的子集
    # 普通解法


def subsets(nums):
    result = [[]]
    for i in nums:
        for j in result[:]:
            x = j[:]
            x.append(i)
            result.append(x)
    return result


nums = [1, 2, 3]
print(subsets(nums))


# %%
# recursion vision
# 所有子集
def subsets(nums):
    lst = []
    result = []
    subsets_recursive_helper(result, lst, nums, 0)
    return result


def subsets_recursive_helper(result, lst, nums, pos):
    result.append(lst[:])
    for i in range(pos, len(nums)):
        lst.append(nums[i])
        subsets_recursive_helper(result, lst, nums, i+1)
        lst.pop()


nums = ['a', 'b', 'c']
print(subsets(nums))


# %%
# 如果有重复的数值
# 排序

def subsets2(nums):
    lst = []
    result = []
    nums.sort()
    subsets_recursive_helper(result, lst, nums, 0)
    return result


def subsets_recursive_helper2(result, lst, nums, pos):
    result.append(lst[:])
    for i in range(pos, len(nums)):

        if nums[i] == nums[i-1]:
            continue

        lst.append(nums[i])
        subsets_recursive_helper2(result, lst, nums, i+1)
        lst.pop()


nums = ['a', 'b', 'c', 'c']
print(subsets2(nums))


# %%
# Permutation 排列组合

def permute(nums):
    perms = [[]]
    for n in nums:
        new_perms = []
        for perm in perms:
            for i in range(len(perm)+1):
                new_perms.append(perm[:1]+[n]+perm[i:])
        perms = new_perms
    return perms


nums = [1, 2, 3]
print(permute(nums))


# %%
# House Robber
def rob(nums):
    n = len(nums)
    dp = [[0]*2 for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1])
        dp[i][1] = nums[i-1]+dp[i-1][0]
    return max(dp[n][0], dp[n][1])


nums = [2, 7, 9, 3, 1]
rob(nums)


# %%
# House Robber
def rob(nums):
    yes, no = 0, 0
    for i in nums:
        no, yes = max(yes, no), i+no
    return max(yes, no)


nums = [2, 7, 9, 3, 1]
rob(nums)


# %%
# 环形房子
# 分两种情况进行判断

# House Robber
def rob(nums):
    if len(nums) == 0:
        return 0

    if len(nums) == 1:
        return nums[0]

    return max(rob_help(nums, 0, len(nums)-1), rob_help(nums, 1, len(nums)))


def rob_help(nums, start, end):
    yes, no = nums[start], 0
    for i in range(start+1, end):
        no, yes = max(yes, no), i+no
    return max(yes, no)


nums = [2, 7, 9, 3, 1]
rob(nums)


# %%

def Tile(n):
    dp = [0]*(n+1)
    dp[0], dp[1], dp[2] = 0, 1, 2
    for i in range(3, n+1):
        dp[i] = dp[i-1]+dp[i-2]

    print(dp)
    return dp[n]


print(Tile(10))


# %%
def minCostClimbingStairs(cost):
    n = len(cost)+1
    dp = [0]*n
    for i in range(2, n):
        dp[i] = min(dp[i - 2] + cost[i - 2], dp[i - 1] + cost[i - 1])

    return dp[n-1]


cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1, 22]
minCostClimbingStairs(cost)


# %%
def numDecodings(s):
    # 判断异常条件
    if s == '' or s[0] == '0':
        return 0
    dp = [1, 1]
    for i in range(2, len(s)+1):
        result = 0
        if 10 <= int(s[i-2:i]) <= 26:
            result += dp[i-2]
        if s[i-1] != '0':
            result += dp[i-1]
        dp.append(result)
    return dp[len(s)]


numDecodings("110")


# %%
# Ex8 Unique Binary Search Trees
# 卡特兰数
def numTrees(n):
    if n <= 2:
        return n
    sol = [0]*(n+1)
    sol[0] = sol[1] = 1
    for i in range(2, n+1):
        for left in range(0, i):
            sol[i] += sol[left]*sol[i-1-left]
    return sol[n]


print(numTrees(3))
print([numTrees(i) for i in range(1, 10)])


# %%
# Ex. 9 Maximum Product Subarray
def maxProduct(nums):
    if len(nums) == 0:
        return 0
    res = maximum = minimum = nums[0]
    for i in range(1, len(nums)):
        maximum = max(maximum*nums[i], minimum*nums[i], nums[i])
        minimum = min(maximum*nums[i], minimum*nums[i], nums[i])
        res = max(maximum, res)

    return res


nums = [2, 3, -2, 4, ]
maxProduct(nums)


# %%
len([2, 3, -2, 4, ])


# %%
def maxProfit(prices):
    if len(prices) < 2:
        return 0
    minPrice = prices[0]
    maxProfit = 0
    for price in prices:
        if price < minPrice:
            minPrice = price
        if price-minPrice > maxProfit:
            maxProfit = price-minPrice

    return maxProfit


prices = [7, 1, 5, 3, 6, 4]
maxProfit(prices)


# %%

def maxProfit(prices):
    if len(prices) < 2:
        return 0
    minPrice = prices[0]
    dp = [0] * len(prices)
    for i in range(len(prices)):
        dp[i] = max(dp[i-1], prices[i]-minPrice)
        minPrice = min(minPrice, prices[i])
    return dp[-1]


prices = [7, 1, 5, 3, 6, 4]
maxProfit(prices)


# %%
def maxProfit2(prices):
    if len(prices) < 2:
        return 0

    maxProfit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            maxProfit += prices[i]-prices[i-1]
    return maxProfit


prices = [7, 1, 5, 3, 6, 4]
maxProfit2(prices)


# %%
def maxProfit2(prices):
    if len(prices) < 2:
        return 0

    maxProfit = 0
    for i in range(1, len(prices)):
        maxProfit += max(0, prices[i]-prices[i-1])
    return maxProfit


prices = [7, 1, 5, 3, 6, 4]
maxProfit2(prices)


# %%
def maxProfit3(prices):
    cash, hold = 0, -prices[0]

    for i in range(1, len(prices)):
        cash, hold = max(cash, hold+prices[i]), max(hold, cash-prices[i])
    return cash


prices = [7, 1, 5, 3, 6, 4]
maxProfit3(prices)


# %%
max([7, 1, 5, 3, 16, 4])


# %%
class Node:
    def __init__(self, value=None, next=None):
        self.value = value
        self.next = next


# %%
# Delete Node
def delete(node):
    print(node.value)
    node.value = node.next.value
    node.next = node.next.next


# %%
def find_last(lst, target):
    if len(lst) == 0:
        return -1
    left, right = 0, len(lst)-1
    while left+1 < right:
        mid = (left+right)//2

        if lst[mid] < target:
            left = mid
        elif lst[mid] > target:
            right = mid
        else:
            left = mid

    if lst[right] == target:
        return right
    if lst[left] == target:
        return left

    return -1


find_last([1, 2, 3, 3, 5], 3)


# %%
# class ListNode:

#   def __init__(self, x):

#     self.val = x

#     self.next = None

def remove_node(lst, target):
    if not lst:
        return lst

    dummy = curr = ListNode(None)
    curr.next = lst

    while curr:
        if curr.next == target:
            curr.next = curr.next.next
        curr = curr.next

    return dummy.next


# %%


# %%
def bi_search_iter(alist, item):

    left, right = 0, len(alist)-1

    while left+1 < right:

        mid = (left+right)//2

        if alist[mid] < item:
            left = mid
        elif alist[mid] > item:
            right = mid
        else:
            left = mid

    if alist[right] == item:
        return right

    if alist[left] == item:
        return left

    return -1


num_list = [1, 2, 7, 7, 7, 8, 9]
print(bi_search_iter(num_list, 7))

print(bi_search_iter(num_list, 4))


# %%
def search(alist):
    if len(alist) == 0:
