# 给一个只包含0和1的数组，找出最长的全是1的子数组。

# Example:

# Input: [1,1,0,1,1,1]

# Output: 3


def find_consecutive_ones(nums):
    local = maxium = 0
    for i in nums:
        local = local+1 if i == 1 else 0
        maxium = max(local,maxium)
    return maxium


nums = [1,1,0,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,1]
result = find_consecutive_ones(nums)
print(result)