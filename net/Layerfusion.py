import torch


class patch_conv(torch.nn.Module):
    def __init__(self, patch_size, patch_n_size, channel, orig_channel=-1) -> None:
        super(patch_conv, self).__init__()
        self.patch_size = patch_size
        self.patch_n_size = patch_n_size
        self.conv_size = patch_size // patch_n_size
        if orig_channel == -1:
            orig_channel = channel
        self.orig_channel = orig_channel
        self.conv = torch.nn.Conv2d(self.orig_channel, channel, [self.conv_size, self.conv_size],
                                    [self.conv_size, self.conv_size])

    def forward(self, x):
        if self.patch_size > self.patch_n_size:
            return self.conv(x)
        else:
            return x


class patch_split(torch.nn.Module):
    def __init__(self, patch_size, img_size) -> None:
        super(patch_split, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.patch_num = max(img_size[0], img_size[1]) // patch_size
        assert len(img_size) == 2, "img dim error"

    def forward(self, x):
        assert x.shape[1] == self.img_size[0] and x.shape[2] == self.img_size[
            1], f"input size{x.shape[1]}*{x.shape[2]} is not equal to img size{self.img_size[0]}*{self.img_size[1]}"
        if self.img_size[0] > self.img_size[1]:
            x = torch.transpose(x, 1, 2)
        x_new = None
        for i in range(self.patch_num):
            patch = x[:, :, i * self.patch_size:(i + 1) * self.patch_size]
            if x_new is None:
                x_new = patch
                x_new = torch.unsqueeze(x_new, 1)
            else:
                x_new = torch.concatenate([x_new, torch.unsqueeze(patch, 1)], dim=1)
        return x_new


class patch_concat(torch.nn.Module):
    def __init__(self, img_size_1, img_size_2, cov_size, chanel1=-1, chanel2=-1):
        super(patch_concat, self).__init__()
        if img_size_1[0] != img_size_1[1]:
            self.ps1 = patch_split(min(img_size_1[0], img_size_1[1]), img_size_1)
            self.pc1 = patch_conv(min(img_size_1[1], img_size_1[0]), cov_size,
                                  max(img_size_1[1], img_size_1[0]) // min(img_size_1[1], img_size_1[0]))
        else:
            self.ps1 = lambda x: x
            self.pc1 = patch_conv(min(img_size_1[1], img_size_1[0]), cov_size,
                                  chanel1)
        if img_size_2[0] != img_size_2[1]:
            self.ps2 = patch_split(min(img_size_2[0], img_size_2[1]), img_size_2)
            self.pc2 = patch_conv(min(img_size_2[1], img_size_2[0]), cov_size,
                                  max(img_size_2[1], img_size_2[0]) // min(img_size_2[1], img_size_2[0]))
        else:
            self.ps2 = lambda x: x
            self.pc2 = patch_conv(min(img_size_2[1], img_size_2[0]), cov_size,
                                  chanel2)
        self.cov_size = cov_size
        self.chanel_out = max(img_size_2[1], img_size_2[0]) // min(img_size_2[1], img_size_2[0]) + max(img_size_1[1],
                                                                                                       img_size_1[
                                                                                                           0]) // min(
            img_size_1[1], img_size_1[0])

    def forward(self, x1, x2):
        y1 = self.pc1(self.ps1(x1))
        y2 = self.pc2(self.ps2(x2))
        assert y1.shape[2] == y2.shape[2] and y1.shape[3] == y2.shape[3], "shape error"
        return torch.concatenate([y1, y2], dim=1), y1, y2







