import requests
import argparse
import json
from datetime import datetime, timedelta

def decrease_date_by_one_day(date_str):
    # Convert the input string to a datetime object
    date_object = datetime.strptime(date_str, '%Y-%m-%d')

    # Decrease the date by one day
    previous_day = date_object - timedelta(days=2)

    # Format the result back to 'YYYY-MM-DD'
    previous_day_str = previous_day.strftime('%Y-%m-%d')

    return previous_day_str
              
def download_preprints_nrecent(date_start, n, metadata_json=None):
    
    """downloads the n most recent preprints from biorxiv and associated metadata
    """
    # storing results in json
    if metadata_json:
        with open(metadata_json, "r") as file:
            metadata = json.load(file)
    
    downloaded = 0
    while downloaded < n:
        
        d_range_dl = 0
        date_end = decrease_date_by_one_day(date_start)
        
        # request the n most recent preprints from biorxiv
        url = f"https://api.biorxiv.org/pubs/biorxiv/{date_end}/{date_start}/0" 
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
                #print(pub_journal, pub_date)
            
                # get the details of the preprint from the doi
                details_url = f"https://api.biorxiv.org/details/biorxiv/{pre_doi}"
                detail_response = requests.get(details_url)
                detail_data = detail_response.json()
                
                
                if "collection" in detail_data:
                    try:
                        preprint_detail = detail_data["collection"][0]
                        authors = preprint_detail["authors"]
                        category = preprint_detail["category"]
                        title = preprint_detail["title"]
                    except IndexError:
                        continue
                else:
                    continue
                
                # store metadata 
                metadata[preprint_id] = {"pub_date": pub_date, "pub_journal": pub_journal, "authors": authors, "category": category, "title": title}
                
                # download abstract
                if response.status_code == 200:
                    with open(f"abstracts/{preprint_id}.txt", "w") as file:
                        file.write(preprint_detail["abstract"])
                        #print(f"Downloaded {title}")
                        downloaded += 1
                        d_range_dl += 1
                else:
                    print(f"Failed to download {title}")
                    
        print(f"Downloaded {d_range_dl} preprints from {date_start} to {date_end}")
        # increment date 
        date_start = date_end
    
    # write metadata back to json file
    with open(metadata_json, "w") as file:
        json.dump(metadata, file, indent=5)
        
if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--date_start", type=str, default="2024-01-01")
    argparser.add_argument("-n", "--nrecent", type=int, default=10)
    argparser.add_argument("-m", "--metadata", type=str)

    args = argparser.parse_args()
    
    download_preprints_nrecent(args.date_start, args.nrecent, args.metadata)
