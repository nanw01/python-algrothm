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






class AdvBST3(AdvBST2):
    def printInorderIterative(self):
        node = self._root
        stack = []
        
        while True:
            while (node is not None):
                stack.append(node)
                node = node._left
            if len(stack) == 0:
                return
            
            node = stack.pop()
            print ('[', node._item, ']', end = " ")
            node = node._right

    # Traversal Methods  
    def print_inorder(self):
        self._print_inorder(self._root)
        print('')

    def _print_inorder(self, node):
        if (node is None):
            return
        self._print_inorder(node._left)
        print ('[', node._item, ']', end = " ")
        self._print_inorder(node._right)


bst = AdvBST3()
numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
for i in numbers:
    bst.add(i)
bst.print_inorder()
bst.printInorderIterative()

