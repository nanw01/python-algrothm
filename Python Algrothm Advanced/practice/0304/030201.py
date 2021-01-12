# ** Ex.1 Subset **

# Given a set of distinct integers, nums, return all possible subsets
def subsets(nums):
    result = [[]]
    for num in nums:
        for element in result[:]:
            x = element[:]
            x.append(num)
            result.append(x)
    return result


nums = ['a', 'b']
print(subsets(nums))
