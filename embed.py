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
def embed(input_ids):
    
    with torch.no_grad():
        embeddings = model(input_ids)[0]
    return embeddings[0]
    

def main():
    parser = argparse.ArgumentParser(description='Remove lines with high alphanumeric percentage from a file.')
    parser.add_argument('-f', '--files_path', type=str, help='Path to directory of files')

    args = parser.parse_args()

    abstracts = glob.glob(args.files_path + '/*.txt')
    
    embeddings = []
 
    for file_path in tqdm(abstracts):
        with open(file_path, 'r') as file:
            file = file.read()
            input_ids = tokenize(file)
            embedding1 = input_ids[:,:512]
            embedding = embed(embedding1)
            embeddings.append(embedding)
    X = torch.tensor(embeddings).float()
    torch.save(X, "17K_abstracts_embeddings.pt")
        

if __name__ == '__main__':
    main()
