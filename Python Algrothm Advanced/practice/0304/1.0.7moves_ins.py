def moves_ins(n, forward):
    if n == 0: 
        return
    moves_ins(n-1, True)
    print("enter ", n) if forward else print("exit  ", n)
    moves_ins(n-1, False)    

moves_ins(3, True)