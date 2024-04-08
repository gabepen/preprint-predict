import torch
import torch.nn as nn
import torch.nn.functional as F
from embed import embed

##?? scale up model?

sample_width = 256
input_size = sample_width*2
# output_size1 = 64
n_embd = 768
head_size = 12 * 6
num_heads = 12 #head_size must be divisible by num_heads
num_blocks = 4
t_dropout = 0.2
num_journals = 208
step_size = 256

batch_size = 128


GPU_available = torch.cuda.is_available()
print(GPU_available)

assert head_size % num_heads == 0


class Head(nn.Module):
  
   def __init__(self, head_size):
       super().__init__()
       self.key = nn.Linear(n_embd, head_size, bias=False)
       self.query = nn.Linear(n_embd, head_size, bias=False)
       self.value = nn.Linear(n_embd, head_size, bias=False)
       self.register_buffer("communication_matrix", torch.ones(input_size,input_size))
       self.communication_matrix = torch.tril(self.communication_matrix)
       self.dropout = nn.Dropout(t_dropout)


   def forward(self, x):
       # Input (batch, 2*num_samples, n_embd)
       # Output (batch, 2*num_samples, head_size)
       k = self.key(x) # (batch, input_size, head_size)
       q = self.query(x)  # (batch, input_size, head_size)
       W = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (batch, input_size, input_size)
       W = W.masked_fill(self.communication_matrix == 0, float('-inf')) # (batch, input_size, input_size)
       W = F.softmax(W, dim=-1)
       W = self.dropout(W)


       v = self.value(x) # (batch, input_size, head_size)
       out = W @ v # (batch, input_size, head_size)
       return out
  
class MultiHead(nn.Module):


   def __init__(self,num_heads,head_size):
       super().__init__()
       self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #this can be parallelized
       self.linear = nn.Linear(head_size*num_heads,n_embd)
       self.dropout = nn.Dropout(t_dropout)

   def forward(self, x):
       x = torch.cat([head(x) for head in self.heads],dim=-1) #(batch,input_size,head_size (global))
       x = self.linear(x) #(batch,input_size,n_embd)
       x = self.dropout(x)
       return x
  
class FeedForward(nn.Module):


   def __init__(self, n_embd):
       super().__init__()
       self.net = nn.Sequential(
           nn.Linear(n_embd, 4 * n_embd),
           nn.ReLU(),
           nn.Linear(4 * n_embd, n_embd),
           nn.Dropout(t_dropout)
       )


   def forward(self, x):
       return self.net(x) #(batch,input_size,n_embd)
  
class Block(nn.Module):


   def __init__(self):
       super().__init__()
       self.multihead = MultiHead(num_heads, head_size // num_heads)
       self.ffwd = FeedForward(n_embd)
       self.ln1 = nn.LayerNorm(n_embd)
       self.ln2 = nn.LayerNorm(n_embd)


   def forward(self, x):
       # input = output = (batch,input_size,n_embd)
       x = x + self.multihead(self.ln1(x))
       x = x + self.ffwd(self.ln2(x))
       return x


class TransformerModel1(nn.Module):

    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Embedding(sample_width*2, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])
        self.multihead = MultiHead(num_heads, head_size // num_heads)
        self.linear0 = nn.Linear(768, n_embd)
        self.linear1 = nn.Linear(2*sample_width*n_embd,sample_width*n_embd//2) #can change the output size of this
        self.ln1 = nn.LayerNorm(2*sample_width*n_embd)

        ##
        self.ln2 = nn.LayerNorm(sample_width*n_embd//2)

        self.ln3 = nn.LayerNorm(500)

        self.linear2 = nn.Linear(sample_width * n_embd//2,500) # hardcoded for now
        self.linear3 = nn.Linear(500,num_journals)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(t_dropout)

    @torch.no_grad()
    def evaluate(self, x, split="test"):

        if split == "train":
            x = embed(x)
            if GPU_available:
                x = x.to("cuda")
            return self(x)
        
        elif split == "test":

            y = torch.zeros(len(x), num_journals)
            for i, example in enumerate(x):
                xx = torch.tensor([example[s:s + input_size] for s in range(0, len(example) - input_size, step_size)])
                yy = torch.zeros(num_journals)
                if GPU_available:
                    yy = yy.to("cuda")
                for istart in range(0, xx.shape[0], batch_size):
                    iend = min(istart + batch_size, xx.shape[0])
                    xx_batch = xx[istart:iend]
                    xx_batch = embed(xx_batch)
                    if GPU_available:
                        xx_batch = xx_batch.to("cuda")
                    out = self(xx_batch) 
                    yy += F.softmax(out, dim=-1).sum(dim=0)

                yy /= xx.shape[0]

                y[i] = yy

            return y


                
    def forward(self, x):
        # X (batch, 2*sample_width, n_chrom)
        # x = self.linear0(x) ##??
        #print(x.shape)
        device = "cuda" if torch.cuda.is_available() else "cpu" # make this global
        pos_embd = self.pos_embedding(torch.arange(sample_width*2).to(device)) # (2*sample_width, n_embd)
        x = x + pos_embd #(batch, 2*sample_width, n_embd) # we possibly want to concatenate this instead of adding

        x = self.blocks(x) #(batch, 2*sample_width, n_embd)
        x = x.reshape(x.shape[0], 2*sample_width*n_embd) #(batch, 2*sample_width*n_embd)

        x = self.ln1(x) #(batch,2*sample_width*n_embd)
        x = self.linear1(x) #(batch, sample_width * n_embd//2)
        x = self.dropout(x)
        x = self.gelu(x)

        x = self.ln2(x)
        x = self.linear2(x) #(batch, 100) #add layernorms?
        x = self.dropout(x)
        x = self.gelu(x)

        x = self.ln3(x)
        x = self.linear3(x) #(batch, num_journals)
        #x = x.reshape(-1) #(batch)
       

        # print(x.shape)
        # x = self.multihead(x) #batch, input_size, head_size
        # x = x.view(batch, input_size*head_size)
        # x = self.linear1(x)
        # x = self.relu(x)
        # x = self.linear2(x)


        return x
    
piece_size = 15
