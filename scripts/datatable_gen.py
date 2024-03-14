import json
import argparse
import glob
import torch

def count_unique_key_values(json_file):
    with open(json_file) as file:
        data = json.load(file)
    
    key_values = {}
    for item in data:
        if data[item]['pub_journal'] in key_values:
            key_values[data[item]['pub_journal']] += 1
        else:
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
            
def get_valid_papers(valid_journals, metadata_json):
    with open(metadata_json) as file:
        data = json.load(file)
    
    valid_papers = []
    for preprint in data.keys():
        if data[preprint]['pub_journal'] in valid_journals:
            valid_papers.append(preprint)
    return valid_papers
    
def one_hot_encode(valid_papers, valid_journals, metadata_json):   
    with open(metadata_json) as file:
        data = json.load(file)
    
    encodings = []
    for preprint in valid_papers:
        encoding = [0] * len(valid_journals)
        encoding[valid_journals.index(data[preprint]['pub_journal'])] = 1
        encodings.append(encoding)
    return encodings

def main():
    parser = argparse.ArgumentParser(description='Remove lines with high alphanumeric percentage from a file.')
    parser.add_argument('-f', '--file_path', type=str, help='Path to json file')
    parser.add_argument('-a', '--abstracts', type=str, help='Path to directory of abstracts')
    

    args = parser.parse_args()

    # count the number of preprints per journal in meta data json 
    unique_key_values = count_unique_key_values(args.file_path)
    journal_counts = []
    for journal in unique_key_values:
        journal_counts.append((journal, unique_key_values[journal]))

    # filter out journals with less than 100 preprints
    pub_count = 0
    valid_journals = []
    for j_count_pair in sorted(journal_counts, key=lambda x: x[1]):
        if j_count_pair[1] >= 100:
            valid_journals.append(j_count_pair[0])
            print(f"{j_count_pair[0]}: {j_count_pair[1]}" )
    input()
    
    valid_papers = get_valid_papers(valid_journals, args.file_path)
    
    encodings = one_hot_encode(valid_papers, valid_journals, args.file_path)
    
    one_hot_tensor = torch.tensor(encodings)
    torch.save(one_hot_tensor, "pub_journal_70k.pt")
    with open('valid_papers_70k.txt', 'w') as file:
        for paper in valid_papers:
            file.write(f"{paper}\n")
    

if __name__ == '__main__':
    main()
