def reverse(nums):
    n = len(nums)
    for i in range(len(nums) // 2):
        nums[i], nums[n - i - 1] = nums[n - i - 1], nums[i]
    print(nums)


nums = []
reverse(nums)

nums = [1]
reverse(nums)

nums = [1, 2]
reverse(nums)

nums = [1, 2, 3]
reverse(nums)
