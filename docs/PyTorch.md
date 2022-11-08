## PyTorch

### What happens during model training
Training a model is an iterative process; in each iteration (called an `epoch`) the model :
- makes a guess about the `output`, 
- calculates the `error` in its guess (`loss`), 
- collects the `derivatives of the error with respect to its parameters`, 
- and `optimizes` these parameters using `gradient descent`.
- `Hyperparameters` are adjustable parameters that let you control the optimization process.
  - e.g. : `batch_size`, `learning_rate`, `epochs`, etc.

### Compute gradients
2 ways:
- add `requires_grad=True` in the Tensor definition
- in place `tensor.requires_grad_(True)`
- Gradient function for a tensor / loss can be accesses by `tensor.grad_fn`


### Freeze layers
2 ways:
- Enclose a set of ops within `no_grad()` scope:
```
with torch.no_grad():
    ...
```
  - Use `tensor.detach()`.


### Computational Graphs
- `autograd` keeps a record of data (tensors) and all executed operations (along with the resulting new tensors) in a directed acyclic graph (DAG) consisting of `Function` objects.
- `Function` : https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
- In the DAG, leaves are the input tensors and root are the output tensors.
- backward pass kicks off when `backward()` is invoked on the root. It involes:
  - invoking `grad_fn`
  - accumulating gradients in the tensor's `grad` attribute.
  - use chain rule to propagate gradients from root to leaf tensors.
- **DAGs are dynamic!** It is recreated after each `backward()` call. This allows control flow to modify shape, size and op at every iteration if needed.
- Gradients are accumulated between runs. Use the optimizer to zero these out between epochs.
```
# 1. create optimizer & loss_fn
optimizer = torch.optim.SGD(model.parameter(), learning_rate=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
# 2. forward pass and estimate loss
y1 = model(X)
loss = loss_fn(y1, y)
# 3. zero existing gradients
optimizer.zero_grad()
# 4. back-propagate the loss
loss.backward()
# 5. perform optimization step
optimizer.step()
```
- Example train test loops : https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation


### Model load and save
Save
- parameters are stored in model's `state_dict`.
- These can be persisted using `torch.save(state_dict, path)`
- To load, create an instance of the same model (because that defines the arch) and then load the params.
- `model.eval()` sets `BN` and `Dropout` to inference mode.
- passing entire model to `torch.save` serializes model arch + `state_dict`.
- Class definition needs to be available since underlying module is Python's `Pickle`.

Load
- `model.load_state_dict(torch.load(PATH))` is used to load a model.
- add `strict=False` argument to ignore missing keys - needed for warmstarting models (e.g. : transfer learning)   

Load to CPU/GPU
- add `map_location=torch.device('cpu')` or `map_location=torch.device('cuda:0')`   

Load to GPU using `to` op
- add `model.to(torch.device('cuda'))` after loading the model.

Checkpoints
```
torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
```
- This way can be used to serialize >1 models and other variables as well.


## NN pipeline

### From scratch
```
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
```

### Simplify backprop and model params using `torch.nn`
```
import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

model = Mnist_Logistic()


for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= p.grad * lr
            model.zero_grad()
```


### Simplify backprop further by using `torch.optim`
```
def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
#print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
```

### Simplify Dataloading
```
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
```

### Modularize by creating `fit()` and `get_data()`
```
#### loss batch ####
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

#### Train ####
import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

#### get data ####
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```


#### Using GPU
- Shift raw tensors to device
- Shift model to device

```
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```