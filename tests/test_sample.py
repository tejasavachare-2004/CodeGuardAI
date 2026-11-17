
import os
import pickle


# 1. SQL Injection
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return query


# 2. Command Injection
def run_command(user_input):
    os.system("echo " + user_input)


# 3. Hardcoded Secret
API_KEY = "sk_test_123456789abcdef"


# 4. Unsafe eval
def calculate(expression):
    return eval(expression)


# 5. Unsafe Deserialization
def load_data(data):
    return pickle.loads(data)


# 6. Weak Hashing
import hashlib

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()
