# 1.0.4  Ex.4 Simplify Path
# Given an absolute path for a file (Unix-style), simplify it.

# For example,

# path = "/home/" => "/home"

# path = "/a/./b/../../c/" => "/c"


def simplifyPath(path):

    if len(path) == 0:
        return path

    lst = path.split('/')[1:]

    result = []
    result = []
    for i in lst:
        if i == '.':
            continue
        if i == '':
            continue
        if i == '..':
            if len(result) != 0:
                result.pop()
        else:
            result.append(i)

    if len(result) == 0:
        return '/'

    return ''.join(['/' + i for i in result])


path = "/home/"
print(simplifyPath(path))
path = "/a/./b/../../c/d/../e/f/g/../"
print(simplifyPath(path))
