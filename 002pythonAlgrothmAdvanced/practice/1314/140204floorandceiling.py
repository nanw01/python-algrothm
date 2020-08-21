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


class AdvBST4(AdvBST3):    
    def floor(self, key):
        return self._floor(self._root, key)
    
    def _floor(self, node, key):
        if (not node):
            return None
        if (key == node._item):
            return node
        if (key < node._item):
            return self._floor(node._left, key)
        t = self._floor(node._right, key)
        if t:
            return t
        return node


bst = AdvBST4()
numbers = [40,20,70,50,10,60,30,80]
for i in numbers:
    bst.add(i)
print(bst.floor(40)._item)
print(bst.floor(44)._item)
print(bst.floor(10)._item)
print(bst.floor(5))
print(bst.floor(100)._item)