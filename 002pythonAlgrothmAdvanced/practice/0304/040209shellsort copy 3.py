def shellSort(arr): 

    gap = int(len(arr)/2)
    while gap > 0: 
        for i in range(gap,len(arr)):
            for j in range(i,0,-gap):
                if arr[j]<arr[j-gap]:
                    arr[j],arr[j-gap] = arr[j-gap],arr[j]
        gap = int(gap/2)

arr = [ 12, 34, 54, 2, 3] 
  

print ("排序前:") 
print(arr), 
shellSort(arr) 
print ("\n排序后:") 
print(arr)


def shellSort2(arr):
    gap = int(len(arr)/2)

    while gap>0:

        for i in range(gap,len(arr)):
            for j in range(i,0,-gap):
                if arr[j]<arr[j-gap]:
                    arr[j],arr[j-gap] = arr[j-gap],arr[j]

        gap = int(gap/2)

print ("排序前:") 
print(arr), 
shellSort2(arr) 
print ("\n排序后:") 
print(arr)