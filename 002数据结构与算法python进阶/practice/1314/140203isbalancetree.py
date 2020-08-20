from BinarySearchTree import BinarySearchTree
from BinarySearchTree import Node




class AdvBST1(BinarySearchTree):
    
    def size(self):
        return self._size(self._root)
    
    def _size(self, node):
        if (not node):
            return 0
        return self._size(node._left) + self._size(node._right) + 1


class AdvBST2(AdvBST1):    
    def maxDepth(self):
        return self._maxDepth(self._root)
    
    def _maxDepth(self, node):
        if (not node):
            return 0
        left_depth = self._maxDepth(node._left)
        right_depth = self._maxDepth(node._right)
        return max(left_depth, right_depth) + 1

class AdvBST3(AdvBST2):    
    def minDepth(self):
        return self._minDepth(self._root)
    
    def _minDepth(self, node):
        if (not node):
            return 0
        left_depth = self._minDepth(node._left)
        right_depth = self._minDepth(node._right)
        return min(left_depth, right_depth) + 1
    
    def isBalanced(self):
        return (self.maxDepth() - self.minDepth()) <= 1



bst = AdvBST3()
numbers = [1,2,3,4,5,6,7,8]
for i in numbers:
    bst.add(i)
bst.print_inorder()
print(bst.isBalanced())





bst = AdvBST3()
numbers = [3,1,5]
for i in numbers:
    bst.add(i)
bst.print_inorder()
print(bst.isBalanced())

