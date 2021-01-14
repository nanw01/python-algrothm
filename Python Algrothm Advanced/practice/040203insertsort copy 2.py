# 插入排序
def insert_sort(arr):
    for i in range(len(arr)):
        preIndex = i-1
        current = arr[i]
        while preIndex >= 0 and arr[preIndex] > current:
            arr[preIndex+1] = arr[preIndex]
            preIndex-=1
        arr[preIndex+1] = current
    return arr



if __name__ == "__main__":
    l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0,16,18,-1,-6,99]
    print(insert_sort(l))
