# 1.0.4  Ex.4 Simplify Path
# Given an absolute path for a file (Unix-style), simplify it.

# For example,

# path = "/home/" => "/home"

# path = "/a/./b/../../c/" => "/c"




def simplifyPath(path):
    lst = []
    splits = path.split("/")
    print(splits)
    for s in splits:
        if s == "":
            continue
        if s == ".":
            continue
            
        if s == "..":
            if len(lst) != 0:
                lst.pop()
        else:
            lst.append(s)
    print(lst)
    result = []
    if len(lst) == 0:
        return "/"
    result = ['/' + i for i in lst]
    return ''.join(result)
    

path = "/home/"
print(simplifyPath(path))
path = "/a/./b/../../c/d/../e/f/g/../"
print(simplifyPath(path))