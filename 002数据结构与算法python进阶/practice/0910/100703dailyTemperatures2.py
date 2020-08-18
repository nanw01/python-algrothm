# ### Ex.3 Daily Temperatures

# Given a list of daily temperatures, produce a list that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.

# For example, given the list temperatures = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].

# Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].



def dailyTemperatures2(temperatures):
    if not temperatures: return []
    result, stack = [0] * len(temperatures), []
    stack.append(0)

    for i in range(1, len(temperatures)):
        while stack:
            prev = stack[-1]
            if temperatures[prev] < temperatures[i]:
                result[prev] = i - prev
                stack.pop()
            else:
                break
        stack.append(i)
    return result


t = [73, 74, 75, 71, 69, 72, 76, 73]
print(dailyTemperatures2(t))