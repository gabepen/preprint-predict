from processing import *
from models import *
import multiprocessing
import concurrent.futures
import torch
import torch.nn as nn
from ast import literal_eval
import sys

save_file = sys.argv[1] if sys.argv[1].endswith(".pth") else sys.argv[1] + ".pth"
data_dir = "./test"
GPU_available = torch.cuda.is_available()
print(GPU_available)
num_files = 50_000
num_epochs = 30
#batch_size = 128
batch_size = 128
train_prop = 0.9
num_estimate = 5000
lr_start = 3e-5
lr_end = lr_start/100

GetMemory()
X = torch.load('/private/groups/corbettlab/gabe/pp_predict/preprint-predict/17K_abstracts_embeddings.pt')


print(X.shape)

GetMemory()

y = torch.load('/private/groups/corbettlab/gabe/pp_predict/preprint-predict/pub_journal.pt')

y = y.argmax(dim=-1)

# Scramble data
torch.manual_seed(random_seed)
ind = int(train_prop * X.shape[0])
idx = torch.randperm(X.shape[0])


X = X[idx]
y = y[idx]

# Split data
ind = int(train_prop * X.shape[0])
X_train, X_test = X[:ind], X[ind:]
y_train, y_test = y[:ind], y[ind:]

GetMemory()
GetTime()

# lr variables
lr = lr_start
lr_factor = (lr_end/lr_start)**(1/(num_epochs - 1))

# Define network
model = TransformerModel1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Define functions for getting random batch and calculating loss
def get_batch(split, num_samples):
    X, y = (X_train, y_train) if split == "train" else (X_test, y_test)
    idx = torch.randperm(X.shape[0])[:num_samples]
    X = X[idx]
    y = y[idx]
    if GPU_available:
        return X.to("cuda"), y.to("cuda")
    else:
        return X, y

@torch.no_grad()
def estimate_loss(num_samples):
    model.eval()
    for split in ["train", "test"]:
        X, y = get_batch(split, num_samples)
        y_pred = torch.zeros((num_samples,num_journals))
        for i in range(0, num_samples, batch_size):
            try:
                X_batch = X[i:i+batch_size]
                y_pred[i:i+batch_size] = model(X_batch)
            except IndexError:
                X_batch = X[i:]
                y_pred[i:] = model(X_batch)

        if GPU_available:
            y_pred = y_pred.to("cuda")                       
        
        loss = criterion(y_pred, y)
        predictions = (y_pred).argmax(dim=-1)


        print()
        print(split, "full set:")
        
    
        num_correct = (predictions == y).sum().item()
        acc = num_correct/num_samples
        print(f"Loss {split}: {loss.item():0.5f}, Accuracy {split}: {acc:0.4f}")

    model.train()

    return acc


if GPU_available:
    print("GPU is available.")
    model = model.to("cuda")
    criterion = criterion.to("cuda")
#    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"
else:
    print("No GPU available. Running on CPU.")

GetMemory()
# estimate_loss(min(num_estimate, X_test.shape[0]))
#Training loop

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb: SIZE"

best_acc = 0
model.train()
for epoch in range(num_epochs):
    print("-----------------------------------------------------------------")
    print(f"Started training on epoch {epoch + 1} of {num_epochs}, learning rate {lr:0.7f}")
    GetTime()
    # Scramble data
    idx = torch.randperm(X_train.shape[0])
    X_train = X_train[idx]
    y_train = y_train[idx]

    for ind in range(0,X_train.shape[0],batch_size):

        #print(torch.cuda.max_memory_allocated(),"and", torch.cuda.memory_allocated())
        try:
            X_batch = X_train[ind:ind+batch_size]
            y_batch = y_train[ind:ind+batch_size]
        except IndexError:
            X_batch = X_train[ind:]
            y_batch = y_train[ind:]

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

    lr *= lr_factor
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    acc = estimate_loss(min(num_estimate, X_test.shape[0]))
    # for child in model.children():
    #     if isinstance(child,nn.Linear):
    #         print("next")
    #         print("weight")
    #         print(child.weight)
    #         print("bias")
    #         print(child.bias)

    if save_file.lower() != "none.pth" and acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), save_file)
