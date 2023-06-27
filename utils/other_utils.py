import os


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def create_dir(dpath):
    is_exist = os.path.exists(dpath)
    if not is_exist:
        os.makedirs(dpath)