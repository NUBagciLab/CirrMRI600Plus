import monai 


def AttentionUnet(spatial_dims, in_channels, out_channels, pretrained=False):
    model = monai.networks.nets.AttentionUnet(spatial_dims, in_channels, out_channels)
    return model


def SwinUniter(img_size, in_channels, out_channels, pretrained=False):
    model = monai.networks.nets.SwinUNETR(img_size, in_channels, out_channels)

def UNet(spatial_dims, in_channels, out_channels):
    model = monai.networks.nets.BasicUNet(spatial_dims, in_channels, out_channels)
    return model

