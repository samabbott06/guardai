import csv
import sys

def modify_csv(input_file, output_file, replacement_string):
    # Open the input and output files
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        # Process the header first
        try:
            header = next(reader)
            # Add a new header for the ID column at the beginning
            new_header = ['id','text','label']
            writer.writerow(new_header)
        except StopIteration:
            print("The CSV file appears to be empty.")
            return

        for i, row in enumerate(reader, start=1):
            # Add line number as the first column
            new_row = [i] + row[:1] + [replacement_string]
            writer.writerow(new_row)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py input_file.csv output_file.csv 'replacement_string'")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    replacement_string = sys.argv[3]

    modify_csv(input_file, output_file, replacement_string)
    print(f"Modified CSV saved to {output_file}")