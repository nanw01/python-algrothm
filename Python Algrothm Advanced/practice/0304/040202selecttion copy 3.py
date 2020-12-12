# 选择排序
def selection_sort(A):
    
    for i in range(len(A)): 
        
        # Find the minimum element in remaining  
        # unsorted array 
        min_idx = i 
        for j in range(i+1, len(A)): 
            if A[min_idx] > A[j]: 
                min_idx = j 
                
        # Swap the found minimum element with  
        # the first element         
        A[i], A[min_idx] = A[min_idx], A[i]
        
    return A


if __name__ == "__main__":
    l = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    print(selection_sort(l))
