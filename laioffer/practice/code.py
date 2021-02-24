# Definition for a binary tree node.
import math


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


lst = [-1, -2, -6, -3, -4, None, None, -7, -8, -5, None]

lst_node = [TreeNode(i) for i in lst]

for i in range(0, int(math.log(len(lst_node), 2) // 1)):
    for j in range(int(math.pow(2, i))):
        print("f:", lst[i+j])

    print()
