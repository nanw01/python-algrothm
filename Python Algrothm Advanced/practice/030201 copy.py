# ** Ex.1 Subset **

# Given a set of distinct integers, nums, return all possible subsets
def subsets(nums):
    res = [[]]
    for num in nums:
        for r in res[:]:
            x = r[:]
            x.append(num)
            res.append(x)

    return res


nums = ['a', 'b']
print(subsets(nums))
