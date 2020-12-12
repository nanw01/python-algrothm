
def search(self, dic, target):
    """
    input: Dictionary dic, int target
    return: int
    """
    # write your solution here
    len = 1
    cur = 0
    while True:

    left,right = cur,len-1

    while left<=right:
        mid = left+(right-left)//2
        if dic.get[mid] is None:
        right -= 1
        elif dic.get[mid] < target:
        left = mid
        elif dic.get[mid] > target:
        right = mid
        else:
        return mid
    
    cur = len-cur
    len = len*2

dic = {1:1,2:2}
