import argparse
import glob

def remove_lines_with_high_nonalphanumeric_percentage(files_path, threshold):
    
    '''working with a directory of bioRxiv abstracts, we want to remove lines with high non-alphanumeric percentage
    '''
    
    abstracts_files = glob.glob(files_path + '/*.txt')
    
    for file_path in abstracts_files:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        filtered_lines = []
        for line in lines:
            nonalphanumeric_count = sum(1 for char in line if not char.isalnum() and char != ' ')
            total_count = len(line)
            nonalphanumeric_percentage = nonalphanumeric_count / total_count

            if nonalphanumeric_percentage <= threshold:
                filtered_lines.append(line.strip())
            else:
                print(f"Removed line: {line}")
                print(f"From file: {file_path}")

        with open(file_path, 'w') as file:
            for line in filtered_lines:
                file.write(line + ' ')
    

def main():
    parser = argparse.ArgumentParser(description='Remove lines with high alphanumeric percentage from a file.')
    parser.add_argument('-f', '--files_path', type=str, help='Path to directory of files')
    parser.add_argument('-t', '--threshold', type=float, help='Threshold for alphanumeric percentage')

    args = parser.parse_args()

    remove_lines_with_high_nonalphanumeric_percentage(args.files_path, args.threshold)

if __name__ == '__main__':
    main()
