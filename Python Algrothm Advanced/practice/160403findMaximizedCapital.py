# ### Ex.3 Manage Your Project (IPO)

# You are given several projects. For each project i, it has a pure profit Pi and a minimum capital of Ci is needed to start the corresponding project. Initially, you have W capital. When you finish a project, you will obtain its pure profit and the profit will be added to your total capital.

# To sum up, pick a list of at most k distinct projects from given projects to maximize your final capital, and output your final maximized capital.

# Input: k=2, W=0, Profits=[1,2,3], Capital=[0,1,1]. 

# Output: 4 

# Explanation: Since your initial capital is 0, you can only start the project indexed 0. After finishing it you will obtain profit 1 and your capital becomes 1. With capital 1, you can either start the project indexed 1 or the project indexed 2. Since you can choose at most 2 projects, you need to finish the project indexed 2 to get the maximum capital. Therefore, output the final maximized capital, which is 0 + 1 + 3 = 4. 




import heapq
def findMaximizedCapital(k, W, Profits, Capital):
    pqCap = []
    pqPro = []
    
    for i in range(len(Profits)):
        heapq.heappush(pqCap, (Capital[i], Profits[i]))
        
    for i in range(k):
        while len(pqCap) != 0 and pqCap[0][0] <= W:
            heapq.heappush(pqPro, -heapq.heappop(pqCap)[1])
            
        if len(pqPro) == 0:
            break
        
        W -= heapq.heappop(pqPro)
    
    return W


k=2
W=0
Profits=[1,2,3]
Capital=[0,1,1]

findMaximizedCapital(k, W, Profits, Capital)