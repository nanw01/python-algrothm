from BinarySearchTree import BinarySearchTree
from BinarySearchTree import Node


class AdvBST1(BinarySearchTree):

    def getIterative(self, key):
        node = self._root
        while (node is not None):
            if key == node._item:
                return node._item
            if key < node._item:
                node = node._left
            else:
                node = node._right
        return None



class AdvBST2(AdvBST1):
    def addIterative(self, value):
        newNode = Node(value)
        if (self._root is None):
            self._root = newNode
            return
        
        current = self._root
        parent = None
        while True:
            parent = current
            if (value == current._item):
                return
            if (value < current._item):
                current = current._left
                if (current is None):
                    parent._left = newNode
                    return
            else:
                current = current._right
                if (current is None):
                    parent._right = newNode
                    return


bst = AdvBST2()
numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
for i in numbers:
    bst.addIterative(i)

bst2 = AdvBST2()
numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
for i in numbers:
    bst2.add(i)
bst.print_inorder()
bst2.print_inorder()
bst.print_preorder()
bst2.print_preorder()