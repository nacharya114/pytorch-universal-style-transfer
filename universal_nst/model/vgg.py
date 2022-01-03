import torch
import torchvision
import torch.nn as nn


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor)
        return x

class VGGAutoEncoder(nn.Module):
    def __init__(self, rep_layer='relu_5_1', vgg_version='vgg19', vgg_path=None, wct_layer=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rep_layer = rep_layer
        self.encoder = nn.Sequential()

        vgg = torchvision.models.vgg19(pretrained=True if not vgg_path else False)
        if vgg_path:
            vgg.load_state_dict(torch.load(vgg_path), strict=False)

        i = 1
        j = 1
        for layer in vgg.features.children():
            if isinstance(layer, nn.Conv2d):
                name = f"conv{i}_{j}"
            elif isinstance(layer, nn.MaxPool2d):
                name = f"maxpool_{i}"
                i += 1
                j = 1

            elif isinstance(layer, nn.ReLU):
                name = f"relu_{i}_{j}"
                j += 1
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.encoder.add_module(name, layer)
            if name == rep_layer: break
                
        self.wct = wct_layer

        self.decoder = nn.Sequential()
        
        for name, layer in list(self.encoder.named_children())[::-1]:
            new_name, new_layer = None, None
            if isinstance(layer, nn.Conv2d):
                new_name = "de" + name
                new_layer = nn.ConvTranspose2d(layer.out_channels, layer.in_channels, kernel_size=layer.kernel_size,stride=layer.stride, padding=layer.padding)
            elif isinstance(layer, nn.MaxPool2d):
                new_name = "upsample_" + name[-1]
                new_layer = Interpolate(scale_factor=2)
            elif isinstance(layer, nn.ReLU):
                new_name = "mir_"+ name
                new_layer = nn.ReLU(inplace=True)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            self.decoder.add_module(new_name, new_layer)
        del vgg

            
    def forward(self, x, style=None):
        enc_x = self.encoder(x)
        if self.wct and style and not self.training:
            enc_style = self.encoder(style)
            enc_x = self.wct(enc_x, enc_style)
        dec_x = self.decoder(enc_x)
        
        if self.training:
            dec_enc = self.encoder(dec_x)
            return dec_x, enc_x, dec_enc
        return dec_x