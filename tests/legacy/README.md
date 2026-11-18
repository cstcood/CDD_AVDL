# Reference implementation

These four files are the released net/ sources, kept here so that
tests/test_equivalence.py can transplant their weights into the refactored
model and compare outputs.

Three mechanical patches were applied; none of them changes the computation:

1. The assertion timm.__version__ == "0.4.5" is removed, because the test
   environment runs timm 1.x.
2. The torch.cuda.amp.autocast decorator on ASTModel.forward is replaced by an
   identity decorator, so the reference runs in full precision on CPU.
3. timm.models.layers is imported from timm.layers and the backbone name
   vit_deit_ is renamed deit_, matching timm 1.x.

The global PatchEmbed monkey patch is left in place; the test restores the
reference patch embedding explicitly because timm 1.x validates the image size.
