import heapq
li1 = [6, 7, 9, 4, 3, 5, 8, 10, 1]
heapq.heapify(li1)
print("The 3 largest numbers in list are : ",end="")
print(heapq.nlargest(3, li1))


print("The 3 smallest numbers in list are : ",end="")
print(heapq.nsmallest(3, li1))


portfolio = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}
]



cheap = heapq.nsmallest(3,portfolio,key=lambda s: s['shares'])
print(cheap)



expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])
print(expensive)