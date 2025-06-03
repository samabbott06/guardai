import threading
import csv
from concurrent.futures import ThreadPoolExecutor

# GLOBALS
# Input file, this is the path to the file that lines will be added from:
INPUT_FILE = 'training_sets/test.csv'

# Path to the target file. This is the file that lines will be added to:
MASTER_FILE = 'training_sets/Master_test.csv'

write_lock = threading.Lock()

def load_data(data_set: str) -> set:
    """
    Load the data from the .csv file.

    Args:
        None

    Returns:
        set: Set of tuples containing the master file data.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(data_set, 'r') as file:
        return set(map(tuple, csv.reader(file)))

def write_line(line: list) -> None:
    """
    Write a line to the master file.

    Args:
        line (list): The line to write.
    
    Returns:
        None
    
    Raises:
        None
    """
    with write_lock, open(MASTER_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(line)

def process_line(line: list, 
                 master_data: set
                 ) -> None:
    """
    Process a line from the test file.

    Args:
        line (list): The line to process.
        master_data (set): Set of tuples containing the master file data.
    
    Returns:
        None
    
    Raises:
        None
    """
    if tuple(line) not in master_data:
        write_line(line)

def main():
    """
    Main function to execute the script.
    """
    master_data = load_data(MASTER_FILE)
    input_data = load_data(INPUT_FILE)
    tpe = ThreadPoolExecutor(max_workers=24)

    with tpe as executor:
        for line in input_data:
            executor.submit(
                process_line, 
                line, 
                master_data
            )

# Entry point
if __name__ == '__main__':
    main()