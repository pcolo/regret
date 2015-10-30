# -*- coding: utf-8 -*-


def div(a, b):
    if b == 0:
        return None
    elif a == 0:
        return 0
    else:
        return a / b

# print(div(3, 5))

liste = [3, 5, 7, 9, 11, 13]
for idx, val in enumerate(liste[2:], 2):
    print idx, "-", val


def factoriel(n=1):
    s = 1
    if n == 0:
        return 1
    for m in range(1, n+1, 1):
        s *= m
    return s

print factoriel(0)

