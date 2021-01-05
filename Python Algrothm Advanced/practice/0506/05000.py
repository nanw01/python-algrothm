def closest(self, array, target):
    """
    input: int[] array, int target
    return: int
    """
    # write your solution here
    if len(array) == 0:
        return -1

    left, right = 0, len(array) - 1

    if target < array[left]:
        return 0
    if target > array[right]:
        return right

    while left < right:
        mid = left + (right - left) // 2

        if target < array[mid]:
            right = mid - 1
        elif target > array[mid]:
            left = mid - 1
