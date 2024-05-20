import random

def split_list_randomly(l):
    random.shuffle(l)
    
    midpoint = len(l) // 2
    
    first_half = l[:midpoint]
    second_half = l[midpoint:]

    return first_half, second_half

first_half, second_half = split_list_randomly([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 3])

print(first_half, len(first_half))
print(second_half, len(second_half))