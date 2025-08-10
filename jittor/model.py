import math
import jittor as jt
from jittor import init
from jittor import nn

def pixel_mse_loss(predictions, targets, pixel_pos):
    jt.flags.use_cuda = 1
    mask = jt.zeros(targets.shape)    # 全0mask
    # 被mask的像素置1
    for i,(h,w) in enumerate(pixel_pos):
        mask[i, :, h, w] = 1.
    # 计算loss
    return nn.mse_loss(predictions*mask, targets*mask)*100000    # 放大loss方便观察

class REDNet10(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, num_layers=5, num_features=64):
        super(REDNet10, self).__init__()
        conv_layers = []
        deconv_layers = []
        conv_layers.append(nn.Sequential(nn.Conv(in_channels, num_features, 3, stride=2, padding=1), nn.ReLU()))
        for i in range((num_layers - 1)):
            conv_layers.append(nn.Sequential(nn.Conv(num_features, num_features, 3, padding=1), nn.ReLU()))
        for i in range((num_layers - 1)):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose(num_features, num_features, 3, padding=1), nn.ReLU()))
        deconv_layers.append(nn.ConvTranspose(num_features, out_channels, 3, stride=2, padding=1, output_padding=1))
        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU()

    def execute(self, x):
        residual = x
        out = self.conv_layers(x)
        out = self.deconv_layers(out)
        out += residual
        out = nn.relu(out)
        return out

class REDNet20(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, num_layers=10, num_features=64):
        super(REDNet20, self).__init__()
        self.num_layers = num_layers
        conv_layers = []
        deconv_layers = []
        conv_layers.append(nn.Sequential(nn.Conv(in_channels, num_features, 3, stride=2, padding=1), nn.ReLU()))
        for i in range((num_layers - 1)):
            conv_layers.append(nn.Sequential(nn.Conv(num_features, num_features, 3, padding=1), nn.ReLU()))
        for i in range((num_layers - 1)):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose(num_features, num_features, 3, padding=1), nn.ReLU()))
        deconv_layers.append(nn.ConvTranspose(num_features, out_channels, 3, stride=2, padding=1, output_padding=1))
        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU()

    def execute(self, x):
        residual = x
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if ((((i + 1) % 2) == 0) and (len(conv_feats) < (math.ceil((self.num_layers / 2)) - 1))):
                conv_feats.append(x)
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (((((i + 1) + self.num_layers) % 2) == 0) and (conv_feats_idx < len(conv_feats))):
                conv_feat = conv_feats[(- (conv_feats_idx + 1))]
                conv_feats_idx += 1
                x = (x + conv_feat)
                x = nn.relu(x)
        x += residual
        x = nn.relu(x)
        return x

class REDNet30(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, num_layers=15, num_features=64):
        super(REDNet30, self).__init__()
        self.num_layers = num_layers
        conv_layers = []
        deconv_layers = []
        conv_layers.append(nn.Sequential(nn.Conv(in_channels, num_features, 3, stride=2, padding=1), nn.ReLU()))
        for i in range((num_layers - 1)):
            conv_layers.append(nn.Sequential(nn.Conv(num_features, num_features, 3, padding=1), nn.ReLU()))
        for i in range((num_layers - 1)):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose(num_features, num_features, 3, padding=1), nn.ReLU()))
        deconv_layers.append(nn.ConvTranspose(num_features, out_channels, 3, stride=2, padding=1, output_padding=1))
        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU()

    def execute(self, x):
        residual = x
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if ((((i + 1) % 2) == 0) and (len(conv_feats) < (math.ceil((self.num_layers / 2)) - 1))):
                conv_feats.append(x)
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (((((i + 1) + self.num_layers) % 2) == 0) and (conv_feats_idx < len(conv_feats))):
                conv_feat = conv_feats[(- (conv_feats_idx + 1))]
                conv_feats_idx += 1
                x = (x + conv_feat)
                x = nn.relu(x)
        x += residual
        x = nn.relu(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(Unet, self).__init__()
        self._block1 = nn.Sequential(nn.Conv(in_channels, 48, 3, stride=1, padding=1), nn.ReLU(), nn.Conv(48, 48, 3, padding=1), nn.ReLU(), nn.Pool(2, op='maximum',stride=2))
        self._block2 = nn.Sequential(nn.Conv(48, 48, 3, stride=1, padding=1), nn.ReLU(), nn.Pool(2, op='maximum',stride=2))
        self._block3 = nn.Sequential(nn.Conv(48, 48, 3, stride=1, padding=1), nn.ReLU(), nn.ConvTranspose(48, 48, 3, stride=2, padding=1, output_padding=1))
        self._block4 = nn.Sequential(nn.Conv(96, 96, 3, stride=1, padding=1), nn.ReLU(), nn.Conv(96, 96, 3, stride=1, padding=1), nn.ReLU(), nn.ConvTranspose(96, 96, 3, stride=2, padding=1, output_padding=1))
        self._block5 = nn.Sequential(nn.Conv(144, 96, 3, stride=1, padding=1), nn.ReLU(), nn.Conv(96, 96, 3, stride=1, padding=1), nn.ReLU(), nn.ConvTranspose(96, 96, 3, stride=2, padding=1, output_padding=1))
        self._block6 = nn.Sequential(nn.Conv((96 + in_channels), 64, 3, stride=1, padding=1), nn.ReLU(), nn.Conv(64, 32, 3, stride=1, padding=1), nn.ReLU(), nn.Conv(32, out_channels, 3, stride=1, padding=1), nn.LeakyReLU(scale=0.1))

    def execute(self, x):
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)
        upsample5 = self._block3(pool5)
        concat5 = jt.contrib.concat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = jt.contrib.concat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = jt.contrib.concat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = jt.contrib.concat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = jt.contrib.concat((upsample1, x), dim=1)
        output = self._block6(concat1)
        return output


if __name__ == '__main__':
    # x = torch.randn(1,3,63,63)
    # model = REDNet30()
    # output1 = model.conv_layers(x) # torch.Size([1, 64, 32, 32])
    # output2 = model.deconv_layers(output1) #
    # print(output1.shape)
    # print(output2.shape)
    #
    # output = model(x)
    # print(x.shape)
    # print(output.shape)

    x = jt.randn(6, 1, 64, 64)
    model = REDNet10(1, 1)
    output = model(x)
    print(x.shape)
    print(output.shape)