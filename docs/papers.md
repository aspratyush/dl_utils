## BackBones
### 1. ResNet
* paper : https://arxiv.org/abs/1512.03385
* visualization : https://pytorch.org/hub/pytorch_vision_resnet/
* code : https://github.com/pytorch/vision/blob/015378434c1e4be778b2d65eb86f227db20bc8bf/torchvision/models/resnet.py

## Action Recognition
### 1. TSN (Temporal Segment Network)
* paper : https://arxiv.org/pdf/1608.00859.pdf
* PyTorch code : https://github.com/yjxiong/tsn-pytorch
* TF1 code : https://github.com/MichiganCOG/M-PACT
* Summary:
    * TODO


### 2. TSM (good)
* ICCV2019 paper : https://arxiv.org/pdf/1811.08383v3.pdf
* PyTorch (original) : https://github.com/mit-han-lab/temporal-shift-module
* TF code : https://github.com/deepmind/deepmind-research/blob/master/mmv/models/tsm_resnet.py
* simpler code : https://github.com/PingchuanMa/Temporal-Shift-Module
* summary:
    * convolution operation consists of shift and multiply-accumulate.
    * shift in the time dimension by ±1 and fold the multiply-accumulate from time dimension to channel dimension.
    * future frames can’t get shifted to the present, use a uni-directional TSM to perform online video understanding.
    * high-efficiency model design by using `partial_shift`+ `residual_shift`.
    * `partial_shift` : shift some temporal features.
        *  if the proportion is too small, the ability of temporal reasoning may not be enough to handle complicated temporal relationships.
        * if too large, the spatial feature learning ability may be hurt.
        * For residual shift, we found that the performance reaches the peak when 1/4 (1/8 foreach direction) of the channels are shifted.
    * `residual_shift` : insert TSM inside residual branch rather than outside so that the activation of the current frame is preserved through identity mapping.
    * 8-frames considered, partial shift employed(1/8, 1/4, 1/2)
    * For each inserted temporal shift module, the temporal receptive field will be enlarged by 2, as if running a convolution with the kernel size of 3 along the temporal dimension.
    * Online methodology:
        * For each frame, save the first 1/8 feature maps of each residual block and cache it in the memory.
        * For the next frame, we replace the first 1/8 of the current feature maps with the cached feature maps.
        * We use the combination of 7/8 current feature maps and 1/8 old feature maps to generate the next layer, and repeat.
    * much better accuracy-computation pareto curve.
    * Training setup (section 5.1)
        * Kinetics : 100 training epochs, initial learning rate 0.01 (decays by 0.1 at epoch 40&80), weight decay 1e-4, batch size 64, and dropout 0.5.
        * other datasets : 50 epochs
        * 'common practice' to fine-tune from Kinetics + freeze BN - check [48], [49]
    * backbones : MobileNet-V2 [36], ResNet-50 [17], ResNext-101 [52] and ResNet-50 + Non-local module [49].


### 3. MoviNets
* Paper : https://arxiv.org/pdf/2103.11511.pdf
* TF code : https://github.com/tensorflow/models/tree/1fa648a753b877f18ca3a1de9bb921c3f024c11d/official/vision/beta/projects/movinet
* Summary:
    * TODO


### 4. Self supervised multimodal versatile networks (S3D-G, ResNet50+TSM, ResNet50x2+TSM)
* Paper : https://arxiv.org/pdf/2006.16228.pdf
* TF Hub code : https://github.com/deepmind/deepmind-research/tree/master/mmv


### 5. TEM (Temporal Excitation Module)
* CVPR 2020 paper : https://arxiv.org/pdf/2004.01398.pdf
* PyTorch code : https://github.com/Phoenix1327/tea-action-recognition
* Summary:
    * video level
    * ME (Motion Excitation) module:
        * similar to SENet [19, 18]
        * short range motion modeling.
        * Instead of adopting the pixel-level optical flow as an additional input modality and separating the training of temporal stream with the spatial stream, ME integrates the motion modeling into the whole spatio-temporal feature learning approach.
        * Feature-level motion representations are firstly calculated between adjacent frames. These motion features are then utilized to produce modulation weights.
        * In this way, the networks are forced to discover and enhance the informative temporal features that capture differentiated information.
        * A portion of channels tends to model the static information related to background scenes; other channels mainly focus on dynamic motion patterns describing the temporal difference.
        * X --Conv(red)--> Xred --Temporal_split--> Diff --> Concat over T segments --> Pool --Conv(exp)--> sigmoid per segment per channel --> motion attentive weights --> X+X.A
    * MTA (multiple temporal aggregation) module:
        * enlarge temporal receptive field.
        * Adopts (2+1)D convolutions, but a group of sub-convolutions replaces the 1D temporal convolution in MTA.
        * The sub-convolutions formulate a hierarchical structure with residual connections between adjacent subsets.
        * When the spatiotemporal features go through the module, the features realize multiple information exchanges with neighboring frames, and the equivalent temporal receptive field is thus increased multiple times to model long-range temporal dynamics.
        * Split channels into 4 fragments
        ```
            X0_i = X_i                                    , i = 1
            X0_i = conv_spa * (conv_t * X_i)              , i = 2
            X0_i = conv_spa * (conv_t * (X_i+X0_i-1))     , i = 3,4
        ```

        * Original Residual Block:
        ```
          X --> 1x1 Cov2D --> ME --> MTA --> 1x1 Conv2D --> + --> Y
            |                                               ^
            |                                               |
            v                                               |
            ------------------------------------------------>
        ```

        * Proposed Residual block:
        ```
          X --> 1x1 Cov2D --> 3x3 Conv2D --> 1x1 Conv2D --> + --> Y
            |                                               ^
            |                                               |
            v                                               |
            ------------------------------------------------>
        ```


### 6. TAM (Temporal Adaptive Module)
* paper ICLR2021 : https://arxiv.org/pdf/2005.06803.pdf
* PyTorch code : https://github.com/liu-zhy/temporal-adaptive-module
* Summary:
    * split video to a location sensitive 'importance map' and a location invariant (also video adaptive) 'aggregation kernel'.
    * local-view branch uses temporal conv. to produce 'importance map' (similar to SENet), global-view branch uses FC layers to produce location invariant kernels for 'temporal aggregation'.
    * First global spatial average pooling converts (C,T,H,W) to (C,T,1,1). This is fed to both local and global branch.
    * Local branch:
        * uses a sequence of temporal convolutional layers with ReLU non-linearity. Kernel size K=3.
        * Conv1D --C channels--> BN --C/beta channels--> Conv1D --C channels--> sigmoid --> importance weights V \in (C x T x 1 x 1) ===> Z = {replicate V spatially} \in (C x T X H x W) . X
    * Global branch:
        * adaptive convolution will be applied in a channel-wise manner (thus ignoring channel correlations).
        * FC --> ReLU --> FC --> softmax


### 7. TDN
* CVPR 2021 paper : https://arxiv.org/pdf/2012.10071v2.pdf
* PyTorch code : https://github.com/MCG-NJU/TDN
* Summary:
    * video-level
    * uses upsampling layer
    * later
    
## Segmentation
### 1. U-Net
- paper : Fully Convolutional Networks for Sematic Segmentation
- reuse pre-trained nets
- upsampling layers in decoder arm (`deconv layers / transposed conv layers`)
- skip connections to improve upsampling

### 2. Dilated Convolutions
- paper : multi-scale context aggregation by dilated convolutions
- `dilated / atrous convs` : increase the receptive field without loss of resolution.
- Last 2 pooling layers from VGG are removed, and subsequent conv layers are replaced with `dilated convs`. `dilation=2` between `pool3` and `pool4`, and `dilation=4` after `pool4`.
- segmentation map is `1/8` the size, interpolated to get same size as input.

### 3. Global Conv Networks (GCN)
- paper : Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network
- GCN : approx. large `k x k` kernels by sum of `1 x k + k x 1` and `k x 1` and `1 x k` convolutions.
- Also uses a Boundary Refinement block.
- ResNet with dilated convs used as encoder decoder.

### 4. Rethinking the Value of Labels for Improving Class-Imbalanced Learning (NeurIPS 2020)
- paper : https://arxiv.org/pdf/2006.07529.pdf
- video : https://youtu.be/XltXZ3OZvyI
- Semi-supervised and self-supervised learning for class imbalanced problem

### 5. MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation (CVPR 2019)
- paper : https://arxiv.org/pdf/1903.01945.pdf
    

## Github Links ###
1. https://github.com/dancelogue : Action Recognition using PyTorch
2. VGGFace2 pretrained : https://github.com/ox-vgg/vgg_face2
    * TF model : https://github.com/WeidiXie/Keras-VGGFace2-ResNet50
