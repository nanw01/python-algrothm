from LinkedList import LinkedList
from LinkedList import Node

class LinkedQueue(object):
    def __init__(self):
        self.head = None
        self.tail = None
        self.count = 0  
        
    def enqueue(self, value):
        new_node = Node(value)

        if self.tail is not None:
            self.tail.next = new_node

        else:
            self.head = new_node

        self.tail = new_node
        self.count += 1

    def dequeue(self):
        if not self.is_empty():
            # point head to next node
            tmp = self.head
            self.head = self.head.next
            print("dequeue sucess")
            self.count -= 1
            return tmp
        else:
            raise ValueError("Empty QUEUE")

    def is_empty(self):
        if self.head is None and self.tail is None:
            return True
        else:
            return False

    def peek(self):
        return self.head.data

        
    def __len__(self):
        return self.count    

    def is_empty1(self):
        return self.count == 0
    
    def print(self):
        node = self.head
        while node:
            print(node.value, end = " ")
            node = node.next
        print('')        




myqueue = LinkedQueue()
print ('size was: ', str(len(myqueue)))
myqueue.enqueue(1)
myqueue.enqueue(2)
myqueue.enqueue(3)
myqueue.enqueue(4)
myqueue.enqueue(5)
print ('size was: ', str(len(myqueue)))
myqueue.print()
print(myqueue.dequeue().value)
print(myqueue.dequeue().value)
print ('size was: ', str(len(myqueue)))
myqueue.print()
myqueue.enqueue(6)
myqueue.enqueue(7)
myqueue.dequeue()
myqueue.dequeue()
print ('size was: ', str(len(myqueue)))
myqueue.print()
myqueue.dequeue()
myqueue.dequeue()
myqueue.dequeue()
print ('size was: ', str(len(myqueue)))
myqueue.print()
#myqueue.dequeue()