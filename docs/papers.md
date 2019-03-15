### U-Net
- paper : Fully Convolutional Networks for Sematic Segmentation
- reuse pre-trained nets
- upsampling layers in decoder arm (`deconv layers / transposed conv layers`)
- skip connections to improve upsampling

### Dilated Convolutions
- paper : multi-scale context aggregation by dilated convolutions
- `dilated / atrous convs` : increase the receptive field without loss of resolution.
- Last 2 pooling layers from VGG are removed, and subsequent conv layers are replaced with `dilated convs`. `dilation=2` between `pool3` and `pool4`, and `dilation=4` after `pool4`.
- segmentation map is `1/8` the size, interpolated to get same size as input.

### Global Conv Networks (GCN)
- paper : Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network
- GCN : approx. large `k x k` kernels by sum of `1 x k + k x 1` and `k x 1` and `1 x k` convolutions.
- Also uses a Boundary Refinement block.
- ResNet with dilated convs used as encoder decoder.
