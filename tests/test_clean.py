
# CodeGuard AI - Clean Security Test
# This file contains safe Python code.
# The Security Agent should ideally find ZERO vulnerabilities.

def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


def greet(name):
    return f"Hello, {name}!"


def divide(a, b):
    if b == 0:
        return "Cannot divide by zero"
    return a / b


def find_user(users, user_id):
    for user in users:
        if user["id"] == user_id:
            return user
    return None


def calculate_total(prices):
    return sum(prices)


def get_user_message(name):
    return f"Welcome, {name}!"


def is_adult(age):
    return age >= 18


def main():
    print(add(10, 20))
    print(greet("Tejas"))
    print(is_adult(20))


if __name__ == "__main__":
    main()
