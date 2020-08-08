# ** Ex.9 Parentheses **

# Given n pairs of parentheses, 
# write a function to generate all combinations of well-formed parentheses.


def generateParenthesis(n):
    def generate(prefix, left, right, parens=[]):
        if right == 0:   parens.append(prefix)
        if left > 0:     generate(prefix + '(', left-1, right)
        if right > left: generate(prefix + ')', left, right-1)
        return parens
    return generate('', n, n)


print(generateParenthesis(4))