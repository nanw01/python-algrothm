class Node:

    def __init__(self, item, left=None, right=None):
        self._item = item
        self._left = left
        self._right = right


class BinarySearchTree:

    def __init__(self, root=None):
        self._root = root
    def get(self, key):
        return self.__get(self._root, key)

    def __get(self, node, key):
        if node is None:
            return None
        if key == node._item:
            return node._item
        if key < node._item:
            return self.__get(node._left, key)
        else:
            return self.__get(node._right, key)

    def add(self, value):
        self._root = self.__add(self._root, value)

    def __add(self, node, value):
        if node is None:
            return Node(value)

        if node._item == value:
            pass
        else:
            if value < node._item:
                node.left = self.__add(node._left, value)
            else:
                node.right = self._add(node._right, value)

        return node

    def remove(self, key):
        self._root = self.__remove(self, self._root, key)

    def __remove(self, node, key):
        if node is None:
            return None
        if key < node._item:
            node._left = self.__remove(node._left, key)
        elif key > node._item:
            node._right = self.__remove(node._right, key)

        else:
            if node._left is None:
                node = node._right
            elif node._right is None:
                node = node._right
            else:
                node._item = self.__get_max(node._left)
                node._left = self.__remove(node._left, node._item)

        return node

    def get_max(self):
        return self.__get_max(self._root)

    def __get_max(self, node):
        if node is None:
            return None
        while node._right is not None:
            node = node._right

        return node
