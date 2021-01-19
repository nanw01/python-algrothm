class Solution(object):
    def kClosest(self, array, target, k):
        n = len(array)
        left, right = 0, len(array)-1

        if target > array[n-1]:
            left, right = n-1, n

        elif target < array[0]:
            left, right = -1, 0

        else:
            while left + 1 < right:
                mid = left + (right-left)//2

                if array[mid] <= target:
                    left = mid
                elif array[mid] > target:
                    right = mid

        counts = 0
        list = []
        if array[left] == target:
            list.append(array[left])
            left -= 1
            counts += 1

        while left >= 0 and right < n and counts < k:

            if target-array[left] <= array[right]-target:
                list.append(array[left])
                left -= 1

            else:
                list.append(array[right])
                right += 1

            counts += 1

        while left < 0 and counts < k:
            list.append(array[right])
            right += 1
            counts += 1

        while right >= n and counts < k:
            list.append(array[left])
            left -= 1
            counts += 1

        return list


if __name__ == "__main__":
    s = Solution()

    arr = [1,  5]
    target = 2
    k = 2

    print(s.kClosest(arr, target, k))
