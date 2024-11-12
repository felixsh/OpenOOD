from math import sqrt

# https://math.stackexchange.com/a/284954
def f(b, c):
    """Given to corr(A,B)=b, corr(A,C)=c compute bound on corr(B,C)=a"""
    x = b * c - sqrt(1 - b**2) * sqrt(1 - c**2)
    print(f'a >= {x:.2f} = f({b}, {c})')


if __name__ == '__main__':
    f(0.9, 0.9)
    f(0.9, 0.8)
    f(0.9, 0.7)
    f(0.9, -0.8)

