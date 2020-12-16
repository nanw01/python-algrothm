from sympy import symbols, diff

x, y = symbols('x y', real=True)

print(diff( x**2 + y**3, y))