# 选择排序
def selection_sort(items):
    
    for i in range(len(items)):
        min = i
        for j in range(i+1,len(items)):
            if items[j]<items[i]:
                min = j
        items[i],items[min] = items[min],items[i]
    

    return items


if __name__ == "__main__":
    l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    print(selection_sort(l))
