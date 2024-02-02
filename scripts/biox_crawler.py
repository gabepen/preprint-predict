import requests
import argparse
import json

def download_preprints_arange(date_range):
    
    """downloads all preprints and assocaited metadata from biorxiv within a given date range
    """
    
    print(date_range)
    url = "https://api.biorxiv.org/details/biorxiv/{}/{}/0".format(date_range[0], date_range[1])
    print(url)
    response = requests.get(url)
    data = response.json()
    if "collection" in data:
        for preprint in data["collection"]:
            title = preprint["title"]
            abstract = preprint["abstract"]
            published = preprint["published"]
    
              
   
def download_preprints_nrecent(n, metadata_json=None):
    
    """downloads the n most recent preprints from biorxiv and associated metadata
    """
    # storing results in json
    if metadata_json:
        with open(metadata_json, "r") as file:
            metadata = json.load(file)
    
    # request the n most recent preprints from biorxiv
    url = f"https://api.biorxiv.org/pubs/biorxiv/{n}/0" 
    response = requests.get(url)
    data = response.json()

    # iterate through the preprints and download the metadata
    if "collection" in data:
        for preprint in data["collection"]:
            
            # get the metadata
            pretitle = preprint["preprint_title"]
            pre_date = preprint["preprint_date"]
            pre_doi = preprint["preprint_doi"]
            pre_platform = preprint["preprint_platform"]
            pub_date = preprint["published_date"]
            pub_journal = preprint["published_journal"]
            
            preprint_id = pre_doi.split("/")[-1]   
            if preprint_id in metadata:
                continue
            print(pub_journal, pub_date)
        
            # get the details of the preprint from the doi
            details_url = f"https://api.biorxiv.org/details/biorxiv/{pre_doi}"
            detail_response = requests.get(details_url)
            detail_data = detail_response.json()
            
            
            if "collection" in detail_data:
                preprint_detail = detail_data["collection"][0]
                authors = preprint_detail["authors"]
                category = preprint_detail["category"]
                title = preprint_detail["title"]
            else:
                continue
            
            # store metadata 
            metadata[preprint_id] = {"pub_date": pub_date, "pub_journal": pub_journal, "authors": authors, "category": category, "title": title}
            
            # download abstract
            if response.status_code == 200:
                with open(f"{preprint_id}.txt", "w") as file:
                    file.write(preprint_detail["abstract"])
                    print(f"Downloaded {title}")
            else:
                print(f"Failed to download {title}")
    
    # write metadata back to json file
    with open(metadata_json, "w") as file:
        json.dump(metadata, file, indent=5)
        
if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--date_range", type=str, default="2020-01-01:2020-01-31")
    argparser.add_argument("-n", "--nrecent", type=int, default=10)
    argparser.add_argument("-m", "--metadata", type=str)

    args = argparser.parse_args()
    
    download_preprints_nrecent(args.nrecent, args.metadata)
