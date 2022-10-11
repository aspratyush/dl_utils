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