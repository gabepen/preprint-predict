from transformers import BertTokenizer, BertModel
import torch
import glob
import argparse
from tqdm import tqdm


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def tokenize(sentence):
    
    # transforms sentence of n word chunks to n x 768 matrix of pretrained embeddings
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    return input_ids

    '''
    
    '''
def embed(input_ids, attention_mask):
    
    with torch.no_grad():
        embeddings = model(input_ids = input_ids, attention_mask = attention_mask)[0]
    return embeddings[0]
    

def main():
    parser = argparse.ArgumentParser(description='Remove lines with high alphanumeric percentage from a file.')
    parser.add_argument('-f', '--files_path', type=str, help='Path to file of abstract DOIs to be embedded')
    parser.add_argument('-e', '--existing_embeddings', type=str, help='Path to existing embeddings file')

    args = parser.parse_args()
    
    # load in previous embeddings dataset
    if args.existing_embeddings:
        print('Loading existing embeddings...')
        eX = torch.load(args.existing_embeddings)

    # generate list of abstract DOIs to be embedded 
    abstracts = []
    with open(args.files_path, 'r') as file:
        lines = file.readlines()
        for l in lines:
            abstracts.append(l.strip())
    
    # iterate over list of DOIs, tokenize, and embed
    embeddings = []
    for file_path in tqdm(abstracts):
        
        # find abstract text through DOI
        with open('abstracts/'+file_path+'.txt', 'r') as file:
            
            # tokenize 
            file = file.read()
            input_ids = tokenize(file)
            tokens = input_ids[:,:512]
            
            # pad tokens to 512
            pad_len = 512 - tokens.shape[1]
            if pad_len > 0:
                padding = torch.Tensor([0] * pad_len).unsqueeze(0)  # Add an extra dimension
                tokens = torch.cat([tokens, padding], dim=1)  # Concatenate along the second dimension
            
            # attention mask and embed
            attention_mask = (tokens != 0).long()
            embedding = embed(tokens.long(), attention_mask)
            embeddings.append(embedding)
    
    # stack embeddings 
    X = torch.stack(embeddings)
    
    # build on previous embeddings tensor
    if args.existing_embeddings:
        combined_embeddings = torch.cat([X, eX], dim=0)
    else:
        combined_embeddings = X
        
    # save embeddings to file 
    torch.save(combined_embeddings, "40k_abstracts_embeddings.pt")
        

if __name__ == '__main__':
    main()
