import csv
import sys

'''
Segments a large csv file into multiple ~97.6 megabyte csv files.
Allows for large datasets to be uploaded with git, as it has a 100 megabyte file limit.
'''

# GLOBALS
FILE_NAME = 'Master_plus_imoxto'
MAX_SIZE = 100000000
FILE_NUM = 1

csv.field_size_limit(sys.maxsize)

def write_row (row: list) -> None:
    """
    Writes a row to the csv file. If the file size exceeds MAX_SIZE, it creates a new file.

    Args:
        row (list): The row to write to the csv file.
    Returns:
        None
    Raises:
        None
    """
    global FILE_NUM

    path = f"training_sets/Master_plus_imoxto/Master_imoxto_{FILE_NUM}.csv"
    with open(path, 'a', encoding='utf-8', newline='') as file:
        if file.tell() > MAX_SIZE:
            FILE_NUM += 1
            write_row(row)
        else:
            writer = csv.writer(file)
            writer.writerow(row)

# Read the csv file and write to new files.
input_file = f"training_sets/Master_plus_imoxto/{FILE_NAME}_Sanatized.csv"
with open(input_file, 'r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file)

        for row in reader:
            write_row(row)