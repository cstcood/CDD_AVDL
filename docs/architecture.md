# Architecture

CDD-AVDL combines three branches: an audio branch built on wav2vec2 and AST, a
visual branch built on OpenGraphAU and MLSTM-FCN, and a convolution-layer
fusion that reuses the three MLSTM-FCN convolution blocks.

The reference implementation hard-codes every intermediate width at the call
site (1190, 7161, 1192, 73, 147, 296, 512). This document shows that all of
them follow from two measured quantities, the audio sequence length and the
visual sequence length, and that the refactored code derives them.

## Notation

    Ta    number of audio frames (wav2vec2, 50 Hz)      paper: 1193
    Tv    number of visual frames (OpenGraphAU, 30 Hz)  paper: 7168
    Ca    audio feature width                           paper: 768
    Cv    visual feature width                          paper: 41
    W     sliding window size                           paper: 10
    S     sliding window stride                         paper: 10

## Audio branch

1. A 1-D convolution with kernel 2 maps Ca = 768 channels to 64 channels and
   shortens the sequence by one frame:

       La = Ta - (kernel - 1) = 1193 - 1 = 1192

2. LayerNorm over the 64 channels.

3. AST with patch size 16 and stride (fstride, tstride) = (10, 10). The patch
   grid is computed by the same formula AST uses internally:

       f_dim = (64 - 16) // 10 + 1 = 5
       t_dim = (La - 16) // 10 + 1 = (1192 - 16) // 10 + 1 = 118
       patches = f_dim * t_dim = 590

   The transformer keeps the two distillation tokens, so the positional
   embedding has 592 rows. The head projects to 8 audio features.

## Visual branch

MLSTM-FCN applies three convolution blocks with kernels 8, 5 and 3 and no
padding, so each block shortens the sequence:

    L1 = Tv - 7 = 7161
    L2 = L1 - 4 = 7157
    L3 = L2 - 2 = 7155

The channel widths are 128, 64 and 32. In parallel, the input is cut into
non-overlapping windows of W = 10 frames, averaged inside each window and fed
to a 5-layer LSTM with 16 hidden units; the last hidden state is concatenated
with the global average of the third convolution block and projected to 4
visual features.

## Convolution-layer fusion (TABLE II)

Each stage patchifies two tensors and concatenates them along the channel axis.

### Sequence patchification

A sequence of shape (B, C, L) with L >= C is cut into N = L // C consecutive
blocks of C samples. Each block is a C x C tile, and a strided convolution with
kernel C // ps and stride C // ps reduces it to ps x ps. Because the tile axis
that carries the block index becomes the channel axis, the convolution maps
N input channels to N output channels:

    N = L // C
    kernel = C // ps
    output = (B, N, ps, ps)

### Map patchification

An already patchified tensor of shape (B, C, ps_prev, ps_prev) is reduced with
a convolution of kernel ps_prev // ps and stride ps_prev // ps, keeping C
channels:

    output = (B, C, ps, ps)

The reference patch_conv only runs its convolution when the kernel is larger
than one; a 1x1 projection is skipped and the tensor passes through unchanged.
Patchify reproduces this with project=False, which is what stage 2 uses.

### The three stages

Stage 1, patch size 32:

    visual block 1   (B, 128, 7161) -> N = 7161 // 128 = 55 patches
    audio sequence   (B,  64, 1192) -> N = 1192 //  64 = 18 patches
    concatenated     (B, 73, 32, 32)
    3x3 convolution  (B, 36, 32, 32)

Stage 2, patch size 32:

    visual block 2   (B, 64, 7157) -> N = 7157 // 64 = 111 patches
    stage-1 map      (B, 36, 32, 32) -> 36 channels, 1x1 kernel, skipped
    concatenated     (B, 147, 32, 32)
    3x3 convolution  (B, 73, 32, 32)

Stage 3, patch size 8:

    visual block 3   (B, 32, 7155) -> N = 7155 // 32 = 223 patches
    stage-2 map      (B, 73, 32, 32) -> 73 channels, 4x4 kernel
    concatenated     (B, 296, 8, 8)
    3x3 convolution  (B, 148, 8, 8)
    3x3 convolution  (B, 8, 8, 8)
    flattened        (B, 512)

The classifier maps 512 features to the class logits.

### General form

With visual block widths (C1, C2, C3), visual block lengths (L1, L2, L3),
audio width Ca and audio length La:

    stage 1 channels = L1 // C1 + La // Ca
    stage 2 channels = L2 // C2 + (stage 1 channels) // 2
    stage 3 channels = L3 // C3 + (stage 2 channels) // 2
    fusion width     = final_channels * ps3 * ps3

For the paper geometry this gives 73, 147, 296 and 512, exactly the numbers the
reference implementation hard-codes.

## Early fusion (Eq. 4)

    F = LogSoftmax( Linear( Norm( Concat(X_M, X_A) + X_C / 2 ) ) )

The released code implements the averaging in the log-probability domain:

    y = log_softmax( (y_fusion + y_features) / 2 )

which is what CDDAVDL.forward returns. The weight is configurable through
model.fusion_weight; 0.5 reproduces the paper.

## Deviations from the reference implementation

1. The reference writes self.relu3 after the fourth convolution. Both modules
   are parameter-free ReLUs, so the refactor names it relu4 without changing
   any value.
2. The reference asserts timm.__version__ == "0.4.5" and monkey-patches
   timm.models.vision_transformer.PatchEmbed globally. The refactor replaces
   the patch embedding on the instance and accepts timm 0.4.x and 1.x names.
3. The reference decorates ASTModel.forward with torch.cuda.amp.autocast,
   which forces half precision on CPU-only machines. The refactor leaves
   precision to the training loop.
4. The reference computes patch_concat.chanel_out incorrectly and never uses
   it. The refactor exposes the correct out_channels.

tests/test_equivalence.py transplants the reference weights into the refactored
model and asserts that both produce identical log-probabilities.
