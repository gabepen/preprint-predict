import argparse
import glob 
import json
import requests
from tqdm import tqdm

def get_metadata(pre_doi):
    
    # get the details of the preprint from the doi
    '''
    details_url = f"https://api.biorxiv.org/details/biorxiv/{pre_doi}"
    detail_response = requests.get(details_url)
    detail_data = detail_response.json()
    '''
    
    details_url = f"https://api.biorxiv.org/pubs/biorxiv/10.1101/{pre_doi}"
    detail_response = requests.get(details_url)
    detail_data = detail_response.json()
    
    if "collection" in detail_data:
        try:
            preprint_detail = detail_data["collection"][0]
            authors = preprint_detail["preprint_authors"]
            category = preprint_detail["preprint_category"]
            title = preprint_detail["preprint_title"]
            pub_journal = preprint_detail["published_journal"]
            pub_date = preprint_detail["published_date"]
        
        except IndexError:
            return None
    else:
        return None
    
    # return metadata 
    return {"pub_date": pub_date, "pub_journal": pub_journal, "authors": authors, "category": category, "title": title}
                

def main():
    '''
        script for collecting missing metadata for preprints 
        
        usage: python regen_metadata.py -f metadata.json -a abstracts/
        
        fills out metadata.json with missing DOI metadata 
    '''
    
    parser = argparse.ArgumentParser(description='Process metadata JSON file')
    parser.add_argument('-f', '--filepath', type=str, help='Path to metadata JSON file')
    parser.add_argument('-a', '--abstracts', type=str, help='Path to directory of abstracts')
    args = parser.parse_args()
    
    # load metadata 
    with open(args.filepath, 'r') as file:
        metadata = json.load(file)
    
    # collect abstract paths
    abstracts = glob.glob(f"{args.abstracts}/*.txt")
    
    # monitoring stats 
    num_new_dois = 0
    total_count = 0
    
    # for each abstract in directory 
    for abstract in tqdm(abstracts):
        
        # get preprint DOI and check if it has metadata 
        pre_doi = abstract.split("/")[-1].split(".")[0]
        if pre_doi in metadata:
            total_count += 1
            continue
        
        # get metadata and store in metadata dictionary
        pre_metadata = get_metadata(pre_doi)
        if pre_metadata:
            total_count += 1
            num_new_dois += 1
            metadata[pre_doi] = pre_metadata
    
    # save updated metadata dictionary to json file 
    with open(args.filepath, 'w') as file:
        json.dump(metadata, file, indent=5)
    
    # output some stats on number of DOIs added to metadata json 
    print(f"Added {num_new_dois} new DOIs to metadata.json for a total of {total_count} DOIs.")

if __name__ == '__main__':
    main()