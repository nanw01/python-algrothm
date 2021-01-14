

from ArrayStack import ArrayStack


def sortStack(s):

    r = ArrayStack()

    while not s.is_empty():
        tmp = s.pop()

        while not r.is_empty() and r.top() > tmp:
            s.push(r.pop())

        r.push(tmp)

    return r


mystack = ArrayStack()
print('size was: ', str(len(mystack)))
mystack.push(3)
mystack.push(1)
mystack.push(4)
mystack.push(2)
mystack.push(5)
mystack.push(6)
mystack.push(9)
mystack.push(8)
mystack.push(7)
mystack.printstack()


r = sortStack(mystack)
r.printstack()
