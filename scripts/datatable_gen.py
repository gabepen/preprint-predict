import json
import argparse
import glob

def count_unique_key_values(json_file):
    with open(json_file) as file:
        data = json.load(file)
    
    key_values = {}
    for item in data:
        if data[item]['pub_journal'] in key_values:
            key_values[data[item]['pub_journal']] += 1
        else :
            key_values[data[item]['pub_journal']] = 1

    return key_values

def gen_data_table(metadata_json, abstract_dir, output_file):
    
    '''Generate data table from metadata json and abstracts
    '''
    
    with open(metadata_json, "r") as file:
        metadata = json.load(file) 
    
    
    abstracts = glob.glob(abstract_dir + '/*.txt')
    
    with open(output_file, 'w') as file:
        
        for abstract in abstracts: 
            
             # read in abstract
            with open(abstract, 'r') as a_file:
                abstract_content = a_file.read()
            
             # get published journal and preprint ID
            pub_journal = metadata[abstract.split('/')[-1].split('.')[0]]['pub_journal']
            doi = abstract.split('/')[-1].split('.')[0]
            
            # write out 
            file.write(f"{doi}\t{pub_journal}\t{abstract_content}\n")
        
def main():
    parser = argparse.ArgumentParser(description='Remove lines with high alphanumeric percentage from a file.')
    parser.add_argument('-f', '--file_path', type=str, help='Path to json file')
    parser.add_argument('-m', '--metadata', type=str, help='Path to metadata json file')   
    parser.add_argument('-a', '--abstracts', type=str, help='Path to directory of abstracts')
    

    args = parser.parse_args()

    unique_key_values = count_unique_key_values(args.file_path)
    journal_counts = []
    for journal in unique_key_values:
        journal_counts.append((journal, unique_key_values[journal]))
    
    pub_count = 0
    for j_count_pair in sorted(journal_counts, key=lambda x: x[1]):
        if j_count_pair[1] >= 10:
            pub_count += j_count_pair[1]
            print(f"{j_count_pair[0]}: {j_count_pair[1]}" )
    print(pub_count)
        
    

if __name__ == '__main__':
    main()
