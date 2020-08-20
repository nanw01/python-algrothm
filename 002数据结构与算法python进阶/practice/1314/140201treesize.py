from BinarySearchTree import BinarySearchTree
from BinarySearchTree import Node




class AdvBST1(BinarySearchTree):
    
    def size(self):
        return self._size(self._root)
    
    def _size(self, node):
        if (not node):
            return 0
        return self._size(node._left) + self._size(node._right) + 1

bst = AdvBST1()
numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
for i in numbers:
    bst.add(i)
bst.print_inorder()
bst.size()