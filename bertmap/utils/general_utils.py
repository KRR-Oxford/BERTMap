import sys
import random

def evenly_divide(start, end, num_splits):
    step = (end - start) / num_splits
    return [start + step * i for i in range(num_splits + 1)]

def random_drop(lst):
    lst.remove(lst[random.randrange(0, len(lst))])
    return lst
    
def exclude_randrange(start, end, exclude):
    result = random.randrange(start, end)
    while result == exclude and end - start > 1:
        result = random.randrange(start, end)
    return result


def uniqify(ls):
    return list(dict.fromkeys(ls))


def swap(a, b):
    temp = a 
    a = b
    b = temp
    return a, b

def size_split(sizes):
    max_range = list(range(sum(sizes)))
    splits = []
    for i in range(len(sizes)):
        start = sizes[i - 1] if i >=1 else 0
        end = sizes[i] + start
        splits.append(max_range[start: end])
    return splits


def batch_split(batch_size, max_num):
    """Split into equal parts of {batch_size} as well as the tail"""
    
    if batch_size > max_num:
        print("Fix the batch size to maximum number.")
        batch_size = max_num
        
    max_range = list(range(max_num))
    num_splits = max_num // batch_size
    num_splits = num_splits - 1 if max_num % batch_size == 0 else num_splits  # for edge case, there will be an empty batch
    splits = []
    for i in range(num_splits + 1):
        start = i * batch_size
        end = min((i + 1) * batch_size, max_num)  # dealing the tail part
        splits.append(max_range[start: end])
    assert len(splits) == num_splits + 1
    return splits
    

def equal_split(num_splits, max_num):
    """Split into equal {num_splits} part as well as the tail"""
    max_range = list(range(max_num))
    interval_range = max_num // num_splits
    splits = []
    for i in range(num_splits + 1):
        start = i * interval_range
        end = min((i + 1) * interval_range, max_num)  # dealing the tail part
        splits.append(max_range[start: end])
    assert len(splits) == num_splits + 1
    return splits


def log_print(log_info, log_path: str):
    """Logging information"""
    print(log_info)
    with open(log_path, 'a+') as f:
        f.write(f'{log_info}\n')
    # flush() is important for printing logs during multiprocessing
    sys.stdout.flush()  