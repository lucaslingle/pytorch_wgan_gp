import torch as tc

class ResBlockUp(tc.nn.Module):
    def __init__(self, channels):
        """
        Implementation of 'Res Block Up', consistent with official wgan-gp cifar10 resnet generator implementation.
        See https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar_resnet.py#L113
        """
        super(ResBlockUp, self).__init__()
        self.residual_stack = tc.nn.Sequential(
            tc.nn.Upsample(mode='nearest', scale_factor=2),
            tc.nn.BatchNorm2d(channels),
            tc.nn.ReLU(),
            tc.nn.Conv2d(channels, channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            tc.nn.BatchNorm2d(channels),
            tc.nn.ReLU(),
            tc.nn.Conv2d(channels, channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
        )
        self.upsample_shortcut = tc.nn.Upsample(mode='nearest', scale_factor=2)

        for m in self.residual_stack:
            if isinstance(m, tc.nn.Conv2d):
                tc.nn.init.kaiming_normal_(m.weight)
                tc.nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = self.residual_stack(x)
        shortcut = self.upsample_shortcut(x)
        out = shortcut + residual
        return out

class Generator(tc.nn.Module):
    def __init__(self, img_height, img_width, img_channels, channels, z_dim):
        super(Generator, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.channels = channels
        self.z_dim = z_dim

        self.linear = tc.nn.Linear(z_dim, channels*4*4)
        self.conv_stack = tc.nn.Sequential(
            ResBlockUp(channels),
            ResBlockUp(channels),
            ResBlockUp(channels),
            tc.nn.BatchNorm2d(channels),
            tc.nn.ReLU(),
            tc.nn.Conv2d(channels, img_channels, kernel_size=(3,3), stride=(1,1)),
            tc.nn.Tanh()
        )

    def forward(self, z):
        lin = self.linear(z)
        x = self.conv_stack(lin.view(-1, self.channels, 4, 4))
        return x


