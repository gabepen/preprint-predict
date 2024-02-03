from transformers import BertTokenizer, BertModel
import torch
import glob
import argparse

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def tokenize(sentence):
    
    # transforms sentence of n word chunks to n x 768 matrix of pretrained embeddings
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    with torch.no_grad():
        embeddings = model(input_ids)[0]
    return embeddings[0]


def main():
    parser = argparse.ArgumentParser(description='Remove lines with high alphanumeric percentage from a file.')
    parser.add_argument('-f', '--files_path', type=str, help='Path to directory of files')

    args = parser.parse_args()

    abstracts = glob.glob(args.files_path + '/*.txt')
    
    for file_path in abstracts
        with open(file_path, 'r') as file:
            file = file.read()
            tokenize(file)
        

if __name__ == '__main__':
    main()
