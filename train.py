from processing import *
from models import *
from embed import embed
import random
import torch
import torch.nn as nn
from ast import literal_eval
import sys




save_file = sys.argv[1] if sys.argv[1].endswith(".pth") else sys.argv[1] + ".pth"
data_dir = "./test"
num_files = 50_000
num_epochs = 30
train_prop = 0.9
num_estimate = 5000
lr_start = 3e-5
lr_end = lr_start/100

# replace these with correct path
path_to_file_x = ""
path_to_file_y = ""

"""
This script assumes that preprints are named

f"{path_to_file_x}0"
f"{path_to_file_x}1"
... 
f"{path_to_file_x}{num_files - 1}"

and that each file contains the embedding integer for each word chunk separated by "\n".

Article categorizations are named 
f"{path_to_file_y}0"
f"{path_to_file_y}1"
...
f"{path_to_file_y}{num_files - 1}"

and that each file contains category number of corresponding article.

"""

# create list of train and test indices
train_indices = []
test_indices = []
for i in range(num_files):

    if random.random() < train_prop: # training example
        with open(path_to_file_x + str(i), "r") as f:
            num_input = len(f.read().split('\n'))  # number of word chunks in file
        
        train_indices.extend([(i, s) for s in range(0, num_input - input_size, step_size)])

    else:  # test example
        test_indices.append(i)



GetMemory()
# X = torch.load('/private/groups/corbettlab/gabe/pp_predict/preprint-predict/17K_abstracts_embeddings.pt')

# print(X.shape)

# GetMemory()

# y = torch.load('/private/groups/corbettlab/gabe/pp_predict/preprint-predict/pub_journal.pt')
# y = y.argmax(dim=-1)

GetTime()

# lr variables
lr = lr_start
lr_factor = (lr_end/lr_start)**(1/(num_epochs - 1))

# Define network
model = TransformerModel1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_factor)

@torch.no_grad()
def estimate_loss(num_samples):

    model.eval()
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    for split in ["train", "test"]:
        
        y_pred = torch.zeros((num_samples,num_journals))
        y_true = torch.zeros((num_samples,)).long()

        for istart in range(0, num_samples, batch_size):

            iend = min(istart + batch_size, num_samples)

            X_batch, y_batch = load_batch(slice(istart, iend), split)

            y_pred[istart:iend] = model.evaluate(X_batch, split=split)
            y_true[istart:iend] = y_batch

        if GPU_available:
            y_pred = y_pred.to("cuda") 
            y_true = y_true.to("cuda")  

        loss = criterion(y_pred, y_true)
        predictions = (y_pred).argmax(dim=-1)

        print(f"{split} evaluation:")
    
        num_correct = (predictions == y_true).sum().item()
        acc = num_correct/num_samples
        print(f"Loss: {loss.item():0.5f}, Accuracy: {acc:0.4f}")

    model.train()

    return acc

def load_batch(sli, split):

    if split == "train":
        X = []
        y = []
        for i, s in train_indices[sli]:

            with open(path_to_file_x + str(i), "r") as f:
                lines = f.read().split("\n")

            # xx = torch.tensor([int(num) for num in lines[s:s + input_size]])
            # xx = embed(xx)
            # print(xx.shape)
            X.append([int(num) for num in lines[s:s + input_size]])

            with open(path_to_file_y + str(i), "r") as f:
                num = int(f.read())
            y.append(num)

        return torch.tensor(X), torch.tensor(y)
    
    elif split == "test":

        X = []
        y = []

        for i in test_indices[sli]:
            with open(path_to_file_x + str(i), "r") as f:
                lines = f.read().split("\n")
            X.append([int(num) for num in lines]) # list of tensors

            with open(path_to_file_y + str(i), "r") as f:
                num = int(f.read())
            y.append(num)

        return X, torch.tensor(y)


if GPU_available:
    print("GPU is available.")
    model = model.to("cuda")
    criterion = criterion.to("cuda")
else:
    print("No GPU available. Running on CPU.")

GetMemory()


#Training loop
best_acc = 0
model.train()
for epoch in range(num_epochs):

    print("-----------------------------------------------------------------")
    print(f"Started training on epoch {epoch + 1} of {num_epochs}, learning rate {optimizer.param_groups[0]['lr']:0.7f}")
    GetTime()

    # Scramble data
    random.shuffle(train_indices)

    for istart in range(0, len(train_indices), batch_size):

        iend = min(istart + batch_size, len(train_indices))

        X_batch, y_batch = load_batch(slice(istart, iend), "train")
        X_batch = embed(X_batch)

        if GPU_available:
            X_batch = X_batch.to("cuda")
            y_batch = y_batch.to("cuda")

        try:
            y_pred = model(X_batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            y_pred = model(X_batch)

    
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    acc = estimate_loss(min(num_estimate, len(test_indices)))

    if save_file.lower() != "none.pth" and acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), save_file)
