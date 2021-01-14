# 插入排序
def insert_sort(items):
    for i in range(1,len(items)):
        for j in range(i,0,-1):
            if items[j]<items[j-1]:
                items[j],items[j-1] = items[j-1],items[j]
    return items



if __name__ == "__main__":
    l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0,16,18,-1,-6,99]
    print(insert_sort(l))
