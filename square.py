"""Calculate the square of a number."""


def square(x):
    return x * x


if __name__ == "__main__":
    n = float(input("Enter a number: "))
    print(f"The square of {n} is {square(n)}")
