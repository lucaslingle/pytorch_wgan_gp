import torch as tc

class OptimizedInitialResBlock(tc.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OptimizedInitialResBlock, self).__init__()
        self.residual_stack = tc.nn.Sequential(
            tc.nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            tc.nn.ReLU(),
            tc.nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            tc.nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.downsample_shortcut = tc.nn.Sequential(
            tc.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            tc.nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )
        for m in self.residual_stack:
            if isinstance(m, tc.nn.Conv2d):
                tc.nn.init.kaiming_normal_(m.weight)
                tc.nn.init.zeros_(m.bias)

        for m in self.downsample_shortcut:
            if isinstance(m, tc.nn.Conv2d):
                tc.nn.init.kaiming_normal_(m.weight)
                tc.nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = self.residual_stack(x)
        shortcut = self.downsample_shortcut(x)
        out = shortcut + residual
        return out


class ResBlock(tc.nn.Module):
    """
    Implementation of 'Res Block/Res Block Down', consistent with official wgan-gp cifar10 resnet critic implementation.
    See https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar_resnet.py#L111

    Note in particular that while the paper says the critic uses no batchnorm, in their implementation is actually does,
    just not in the first layer.
    """
    def __init__(self, channels, down=False):
        super(ResBlock, self).__init__()
        self.residual_stack = tc.nn.Sequential(
            tc.nn.BatchNorm2d(channels),
            tc.nn.ReLU(),
            tc.nn.Conv2d(channels, channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            tc.nn.BatchNorm2d(channels),
            tc.nn.ReLU(),
            tc.nn.Conv2d(channels, channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )
        self.downsample_shortcut = tc.nn.Sequential(
            tc.nn.Conv2d(channels, channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )
        self.down = down

        for m in self.residual_stack:
            if isinstance(m, tc.nn.Conv2d):
                tc.nn.init.kaiming_normal_(m.weight)
                tc.nn.init.zeros_(m.bias)

        for m in self.downsample_shortcut:
            if isinstance(m, tc.nn.Conv2d):
                tc.nn.init.kaiming_normal_(m.weight)
                tc.nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = self.residual_stack(x)
        shortcut = self.downsample_shortcut(x)
        out = shortcut + residual
        if self.down:
            out = tc.nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))(out)
        return out


class Critic(tc.nn.Module):
    def __init__(self, img_height, img_width, img_channels, channels):
        super(Critic, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.channels = channels

        self.conv_stack = tc.nn.Sequential(
            OptimizedInitialResBlock(img_channels, channels),
            ResBlock(channels, down=True),
            ResBlock(channels, down=False),
            ResBlock(channels, down=False),
            tc.nn.ReLU()
        )
        self.linear = tc.nn.Linear(channels, 1)

    def forward(self, x):
        spatial_features = self.conv_stack(x)
        features = spatial_features.mean(dim=(2,3))
        critic_output = self.linear(features)
        return critic_output

