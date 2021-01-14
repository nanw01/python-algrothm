# Linked List
class Node(object):

    def __init__(self, value=None, next=None):
        self.value = value
        self.next = next


class LinkedList:
    def __init__(self):
        self.head = Node(0)
        self.length = 0
        self.tail = None

    def peek(self):
        pass

    def get_first(self):
        return self.head.next

    def get_last(self):
        # temp = self.head
        # for _ in range(self.length):
        #     temp = temp.next
        # return temp
        return self.tail

    def get(self, index):
        temp = self.head
        for _ in range(index):
            temp = temp.next
        return temp

    def add_first(self, value):
        self.head.next = Node(value=value, next=self.head.next)
        self.length +=1

    def add_last(self, value):
        newTail = Node(value)

        if self.tail:

            self.tail.next = newTail
            self.tail = newTail
            self.length +=1

        else:
            self.tail = newTail

    def add(self, index, value):

        indexNode = self.head
        for _ in range(index):
            indexNode = indexNode.next

        newNode = indexNode
        indexNode.value = value
        indexNode.next = newNode

        self.length +=1

    def remove_first(self):
        self.head.next = self.head.next.next
        self.length -=1

    def remove_last(self):

        new_tail = self.head
        for _ in range(self.length-1):
            new_tail = new_tail.next
        new_tail.next = None
        self.tail = new_tail
        self.length -=1

    def remove(self, index):

        indexNode = self.head
        for _ in range(index):
            indexNode = indexNode.next
        indexNode.value = indexNode.next.value
        indexNode.next = indexNode.next.next

        self.length -=1

    def printlist(self):
        
        temp = self.head
        for _ in range(self.length):
            print(temp.value, end=' ')
            temp = temp.next
