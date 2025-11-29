import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(
        self, 
        image_dim: int, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        non_linearity: nn.Module,
        downsample: bool = True
    ):
        super().__init__()
        self.image_dim = image_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=kernel_size, 
            padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.non_linearity = non_linearity()
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels, 
            out_channels=self.out_channels, 
            kernel_size=kernel_size, 
            padding=1
        )

        if downsample:
            self.identity = nn.Sequential(
                self.pool,
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    padding=1
                )
            )
        else:
            self.identity = nn.Identity()

    def forward(self, x: torch.tensor):
        if x.shape[-1] != self.image_dim and x.shape[-2] != self.image_dim:
            raise ValueError(f"Tensor of invalid size: {x.shape} passed into Encoder block")
        
        xres = self.identity(x)
        xout1 = self.bn(self.non_linearity(self.conv1(x)))
        xout2 = self.pool(self.non_linearity(self.conv2(xout1)))
        return xout2 + xres
    

class DecoderBlock(nn.Module):
    def __init__(
        self, 
        image_dim: int, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        non_linearity: nn.Module,
        upsample: bool = True
    ):
        super().__init__()
        self.image_dim = image_dim
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.kernel_size = kernel_size

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=1
        )
        self.non_linearity = non_linearity()
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=1
        )

        if upsample:
            self.identity = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    padding=1
                )
            )
        else:
            self.identity = nn.Identity()

    def forward(self, x: torch.tensor, skip_channel: torch.tensor = None):
        if x.shape[-1] != self.image_dim or x.shape[-2] != self.image_dim:
            raise ValueError(f"x with invalid shape: {x.shape} provided, expected last 2 dimensions: {self.image_dim}")

        if skip_channel is not None:
            if skip_channel.shape[-1] != self.image_dim or skip_channel.shape[-2] != self.image_dim:
                raise ValueError(f"Skip channel with invalid shape: {skip_channel.shape} provided, expected last 2 dimensions: {self.image_dim}")

            try:
                assert x.shape == skip_channel.shape
            except AssertionError:
                print("Shapes of input tensor and skip channel given to Decoder block are not same.")
                print(f"Shape of x: {x.shape}, Shape of skip channel: {skip_channel.shape}")

            x = torch.cat((x, skip_channel), axis=1)

        xres = self.identity(x)
        x = self.upsample(x)
        xout1 = self.bn(self.non_linearity(self.conv1(x)))
        xout2 = self.non_linearity(self.conv2(xout1))
        return xout2 + xres



class UNet(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        channels: list[int],
        skip_channels: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channels = channels

        self.encoder = nn.ModuleList()
        current_img_dim = self.input_dim
        prev_channel = 1
        for i in range(len(channels)):
            self.encoder.append(
                EncoderBlock(
                    image_dim=current_img_dim,
                    in_channels=prev_channel,
                    out_channels=channels[i],
                    kernel_size=3,
                    non_linearity=nn.ReLU,
                    downsample=True
                )
            )
            current_img_dim /= 2
            prev_channel = channels[i]

        self.decoder = nn.ModuleList()
        channels.reverse()
        channels = channels[1:]
        channels.append(1)
        for i in range(len(channels)):
            self.decoder.append(
                DecoderBlock(
                    image_dim=current_img_dim,
                    in_channels=prev_channel if not skip_channels else prev_channel*2,
                    out_channels=channels[i],
                    kernel_size=3,
                    non_linearity=nn.ReLU,
                    upsample=True
                )
            )
            current_img_dim *= 2
            prev_channel = channels[i]

        del prev_channel, current_img_dim

    def forward(self, x: torch.tensor):
        if x.shape[-1] != self.input_dim or x.shape[-2] != self.input_dim:
            raise ValueError(f"Expected last 2 dimensions of x.shape: {self.input_dim}, recieved: {x.shape}") 
        
        xout = x
        layer_outputs = []

        for encoder_block in self.encoder:
            xout = encoder_block(xout)
            layer_outputs.append(xout)

        layer_outputs.reverse()
        for i, decoder_block in enumerate(self.decoder):
            xout = decoder_block(xout, layer_outputs[i])

        return xout