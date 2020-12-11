# 冒泡排序
def bubble_sorted(list,reverse=False):

    for i in range(len(list)):
        for j in range(len(list)-i-1):
            if list[j]<list[j+1]:
                list[j],list[j+1] = list[j+1],list[j]

    if reverse is not True:
        list.reverse()

    return list

    
if __name__ == "__main__":
    
    l = [8, 3, 5, 7, 9, 2, 4, 6, 8, 2]
    l = bubble_sorted(l, False)
    print(l)
    l = bubble_sorted(l, True)
    print(l)