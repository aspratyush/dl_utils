## Pytorch

### Initialization
- PyTorch is apparently known to have non-standard initialization.
- Relevant links:
  - https://github.com/pytorch/pytorch/issues/18182
  - https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
  - https://discuss.pytorch.org/t/same-implementation-different-results-between-keras-and-pytorch-lstm/39146/4
- PyTorch reference models force ‘standard’ initializers to overcome the stupid default `sqrt(5)` initializer.
- https://github.com/pytorch/vision/blob/a75fdd4180683f7953d97ebbcc92d24682690f96/torchvision/models/resnet.py#L160-L175

### Good links
- PyTorch to Keras syntax lookup : https://adamoudad.github.io/posts/keras_torch_comparison/syntax/
- Github tutorial : https://github.com/Andrew-Ng-s-number-one-fan/PyTorch-Keras
