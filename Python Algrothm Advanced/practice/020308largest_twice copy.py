# 给定一个数组，数组里有一个数组有且只有一个最大数，
# 判断这个最大数是否是其他数的两倍或更大。
# 如果存在这个数，则返回其index，否则返回-1。

# 思路，找到最大数跟第二最大数

def largest_twice(nums):
    
    maxium=second=idx=0
    for i in range(len(nums)):
        if maxium <nums[i]:
            second = maxium
            maxium = nums[i]
            idx = i
        elif second< nums[i]:
            second=nums[i]

    return idx if(maxium>=second*2) else -1




nums = [1, 2,3,8,3,2,1]
result = largest_twice(nums)
print(result)