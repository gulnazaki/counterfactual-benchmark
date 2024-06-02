import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from models.utils import flatten_list, continuous_feature_map
from models.gans import CondWGAN
from datasets.transforms import get_attribute_ids
from datasets.adni.dataset import bin_array, ordinal_array


class BasicBlock(nn.Module):
    '''
    Basic block for handling the parameters (BasicBlock).
    '''
    def __init__(self, args):
        super(BasicBlock, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
        # Input size
        self.H, self.W = self.args['input_size']
        # Number of filters
        self.IF = self.args['IF']
        self.OF = self.args['OF']
        # Non-linear function to apply (ReLU or LeakyReLU)
        self.nonlinearity = self.args['nonlinearity']
        # Normalization layer
        self.normalization = self.args['normalization']
        if self.normalization == 'batchnorm':
            self.normlayer = lambda x1,x2,x3: nn.BatchNorm2d(x1)
            # self.normlayer = lambda x: nn.BatchNorm2d(x)
        elif self.normalization == 'layernorm':
            self.normlayer = lambda x1,x2,x3: nn.LayerNorm([x1,x2,x3])
            # self.normlayer = lambda x: nn.LayerNorm([x, self.H, self.W])
        else:
            self.normlayer = lambda x1,x2,x3: nn.Identity()


class ConvBlock(BasicBlock):
    '''Convolutional block. A module is composed of two.'''
    def __init__(self, args) -> None:
        super().__init__(args)

        self.block = nn.Sequential(
            nn.Conv2d(self.IF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            self.normlayer(self.OF, self.H, self.W),
            self.nonlinearity,
        )

    def forward(self, x):
        return self.block(x)


class DCB(BasicBlock):
    '''
    Downsampling Convolutional Block (DCB).
    '''
    def __init__(self, args):
        super().__init__(args)

        args['IF'] = self.IF
        args['OF'] = self.OF
        block1 = ConvBlock(args)
        args['IF'] = self.OF
        args['OF'] = self.OF
        block2 = ConvBlock(args)

        self.model = nn.Sequential(
            block1,
            block2,
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        y = self.model(x)

        return y


class UCB(BasicBlock):
    '''
    Upsampling Convolutional Block (UCB).
    '''
    def __init__(self, args):
        super().__init__(args)

        args['IF'] = 2*self.OF
        args['OF'] = self.OF
        block1 = ConvBlock(args)
        args['IF'] = self.OF
        args['OF'] = self.OF
        block2 = ConvBlock(args)

        self.step1 = nn.ConvTranspose2d(
            self.IF, self.OF, self.kernel_size, stride=2, padding=1, output_padding=1, bias=False)
        # TODO: Inverted nonlinearity and normlayer to have the corrected version
        self.step2 = nn.Sequential(
            block1,
            block2,
        )

        # self.step2 = nn.Sequential(block1,block2)

        # args['IF'] = self.IF
        # args['OF'] = self.OF
        # self.step1 = DeConvBlock(args)

    def forward(self, x1, x2):
        x1 = self.step1(x1)
        x = torch.cat((x1, x2), dim=1)
        y = self.step2(x)

        return y


class RandomFourierFeatures:
    def __init__(self, embedding_size: int = 5, features_size: int  = 1):
        scale = 4
        self.embedding_size = embedding_size
        self.features_size = features_size
        torch.manual_seed(1234)
        self.W = torch.randn((features_size, embedding_size)) * scale

    def __call__(self, x):
        assert x.shape[-1] == self.W.shape[0]
        dvc = x.get_device()
        self.W = self.W.to(dvc)
        x_proj = torch.matmul(x, self.W) * 2 * np.pi # [bs,features,embedding_size]
        x_fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x_fourier


class TransformerXia(nn.Module):
    '''
    Combination of fully connected layers with input vectors.
    '''
    def __init__(self, args):
        super(TransformerXia, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
        # Number of filters
        self.IF = self.args['IF']
        self.OF = self.args['OF']
        # Latent space and disease vectors dimension
        self.latent_dim = self.args['latent_dim']
        # Size of smallest image
        self.image_size = self.args['image_size']

        # TODO pass from config
        self.attr_size_dict = {
            "brain_vol": 20,
            "vent_vol": 20,
            "slice": 20
        }
        self.attr_size_dict_input = {
            "brain_vol": 1,
            "vent_vol": 1,
            "slice": 1
        }

        self.attribute_ids = get_attribute_ids(self.attr_size_dict_input)

        # # TODO add extra layers + forward code
        # self.extra_latent = self.args['extra_latent']

        # TODO pass this from config
        self.enc_vec_size = 2*sum(self.attr_size_dict.values())


        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(self.IF, self.OF, self.kernel_size, stride=1, padding=1, bias=False),
            nn.LayerNorm([self.OF, self.image_size, self.image_size]),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            # 15 * 15 * 32 = 7200
            nn.Linear(self.image_size*self.image_size*self.OF, self.latent_dim),
            nn.Sigmoid(),
            nn.LayerNorm(self.latent_dim)
        )

        # Transform a vector through a NN
        self.trans = nn.Sequential(
            # nn.Linear(self.latent_dim + self.num_classes, self.latent_dim),
            # +ve and -ve ordinal encoding
            # Ordinal encoding (OE1)
            nn.Linear(self.latent_dim + self.enc_vec_size, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.latent_dim)
        )

        # Decoder
        self.dec = nn.Sequential(
            # One variable version
            nn.Linear(self.latent_dim, self.image_size*self.image_size*self.OF),
            # Two variable version
            # nn.Linear(self.latent_dim + self.num_classes, self.image_size*self.image_size*self.OF),
            # nn.Linear(self.latent_dim + 2*self.enc_half_size, self.image_size*self.image_size*self.OF),
            nn.ReLU(inplace=True)
        )

        self.encoding_operation = RandomFourierFeatures(list(self.attr_size_dict.values())[0])


    def attr_embedding_fn(self, attr_name, attr, size):
        if attr_name == 'slice':
            attr /= size
        return self.encoding_operation(attr)


    def forward(self, x, cond, given_z=None):
        attrs = []
        for attr_name, size in self.attr_size_dict.items():
            attr = cond[:, self.attribute_ids[attr_name]]
            attr = self.attr_embedding_fn(attr_name, attr, size)
            attrs.append(attr)

        z = self.enc(x) if given_z is None else given_z

        z_cat = torch.concat((z, *attrs), dim=1)
        # z_cat = torch.cat((z, cond), dim=1)
        z2 = self.trans(z_cat)
        out = self.dec(z2)

        out = torch.reshape(out, (x.size(0), self.OF, x.size(2), x.size(3)))

        return out, z


class GeneratorXia(nn.Module):
    '''
    This structure follows the paper design in 1920.02620 and not exactly the
    github code at https://github.com/xiat0616/BrainAgeing. There are differences.
    '''
    def __init__(self, args):
        super(GeneratorXia, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
        # Input shape
        self.input_shape = np.asarray(self.args['input_shape'])
        # number of features to start with
        self.num_feat = args['initial_filters']
        # Latent space and disease vectors dimension
        self.latent_dim = self.args['latent_space_dim']
        self.n_channels = self.args['n_channels']
        self.use_sigmoid = self.args['use_sigmoid']
        self.normalization = self.args['gen_params']['norm']

        if self.args['gen_params']['activation'] == 'lrelu':
            self.nonlinearity = nn.LeakyReLU(0.2, inplace=True)
        elif self.args['gen_params']['activation'] == 'relu':
            self.nonlinearity = nn.ReLU(inplace=True)
        elif self.args['gen_params']['activation'] == 'gelu':
            self.nonlinearity = nn.GELU()
        elif self.args['gen_params']['activation'] == 'silu':
            self.nonlinearity = nn.SiLU(inplace=True)
        elif self.args['gen_params']['activation'] == 'mish':
            self.nonlinearity = nn.Mish(inplace=True)
        else:
            raise ValueError("Activation function '{}' not implemented.".format(
                self.args['gen_params']['activation']))

        # Encoding path
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            # nn.BatchNorm2d(self.num_feat),
            nn.LayerNorm([self.num_feat,self.input_shape[0],self.input_shape[1]]),
        )
        self.enc2 = DCB({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.enc3 = DCB({
            'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat*2})
        self.enc4 = DCB({
            'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat*4})
        self.enc5 = DCB({
            'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*8})

        # Transformer
        self.trans = TransformerXia(
            {'kernel_size': self.kernel_size,
            'IF': self.num_feat*8, 'OF': self.num_feat,
            'latent_dim': self.latent_dim,
            'image_size': self.input_shape[0] // 16,
            # 'encoding': self.args['gen_params']['encoding'],
            # 'encoding': 'both',
            'extra_latent': False}
        )

        # Decoding path
        self.dec4 = UCB({
            'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*(8+1), 'OF': self.num_feat*4})
        self.dec3 = UCB({
            'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*2})
        self.dec2 = UCB({
            'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat})
        self.dec1 = UCB({
            'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})

        self.dec0 = nn.Sequential(
            nn.Conv2d(self.num_feat, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(self.num_feat),
            nn.LayerNorm([self.num_feat, self.input_shape[0], self.input_shape[1]]),
            self.nonlinearity,
            nn.Conv2d(
                self.num_feat, self.n_channels, self.kernel_size, stride=1, padding=1, bias=False)
        )

        if self.use_sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x, cond, given_z=None):
        '''Call function.'''
        # Encoding path
        enc1 = x
        enc2 = self.enc1(enc1)
        enc3 = self.enc2(enc2)
        enc4 = self.enc3(enc3)
        enc5 = self.enc4(enc4)
        enc6 = self.enc5(enc5)

        # Transformer
        aux, z = self.trans(enc6, cond, given_z)

        # Decoding path
        dec5 = torch.cat((aux, enc6), dim=1)
        dec4 = self.dec4(dec5, enc5)
        dec3 = self.dec3(dec4, enc4)
        dec2 = self.dec2(dec3, enc3)
        dec1 = self.dec1(dec2, enc2)

        _map = self.dec0(dec1)

        # Add mapping to input image
        output = _map + x
        output = self.activation(output)

        return output, _map, z



class DiscriminatorXia(nn.Module):
    '''A simpler version than GeneratorXia with fewer layers.'''
    def __init__(self, args):
        super(DiscriminatorXia, self).__init__()

        self.args = args

        # Kernel size NxM
        self.kernel_size = self.args['kernel_size']
        self.input_shape = np.asarray(self.args['input_shape'])
        # number of features to start with
        self.num_feat = args['initial_filters']
        # Latent space and disease vectors dimension
        self.latent_dim = self.args['latent_space_dim']
        self.normalization = self.args['discr_params']['norm']

        if self.args['discr_params']['activation'] == 'lrelu':
            self.nonlinearity = nn.LeakyReLU(0.2, inplace=True)
        elif self.args['discr_params']['activation'] == 'relu':
            self.nonlinearity = nn.ReLU(inplace=True)
        elif self.args['discr_params']['activation'] == 'gelu':
            self.nonlinearity = nn.GELU()
        elif self.args['discr_params']['activation'] == 'silu':
            self.nonlinearity = nn.SiLU(inplace=True)
        elif self.args['discr_params']['activation'] == 'mish':
            self.nonlinearity = nn.Mish(inplace=True)
        else:
            raise ValueError("Activation function '{}' not implemented.".format(
                self.args['discr_params']['activation']))

        # Encoding path. Shapes - (batch size, filters, rows, cols)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, self.num_feat, self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            # nn.BatchNorm2d(self.num_feat),
            nn.LayerNorm([self.num_feat,self.input_shape[0],self.input_shape[1]]),
        )
        self.enc2 = DCB({'input_size': self.input_shape, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat})
        self.enc3 = DCB({'input_size': self.input_shape//2, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat, 'OF': self.num_feat*2})
        self.enc4 = DCB({'input_size': self.input_shape//4, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*2, 'OF': self.num_feat*4})
        self.enc5 = DCB({'input_size': self.input_shape//8, 'nonlinearity': self.nonlinearity,
            'normalization': self.normalization, 'kernel_size': self.kernel_size,
            'IF': self.num_feat*4, 'OF': self.num_feat*8})

        # Divide image size by 2 and multiply feature number by 2
        # Output has dim (BS, 192, 12, 12)
        self.encoder = nn.Sequential(
            self.enc1, self.enc2, self.enc3, self.enc4, self.enc5)

        # Transformer
        self.trans = TransformerXia(
            {'kernel_size': self.kernel_size,
            'IF': self.num_feat*8, 'OF': self.num_feat,
            'latent_dim': self.latent_dim,
            'image_size': self.input_shape[0] // 16,
            # 'encoding': self.args['discr_params']['encoding'],
            # 'encoding': 'positive',
            'extra_latent': True}
        )

        # Judge
        self.judge = nn.Sequential(
            # nn.Dropout(p=0.5),
            # (BS, 256+32, 8, 8)
            nn.Conv2d(self.num_feat*(8+1), self.num_feat*8,
                      self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            nn.Conv2d(self.num_feat*8, self.num_feat*8,
                      self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            nn.Conv2d(self.num_feat*8, self.num_feat*8,
                      self.kernel_size, stride=1, padding=1, bias=False),
            self.nonlinearity,
            # (BS, 256, 8, 8)
            nn.Conv2d(self.num_feat*8, 1,
                      self.kernel_size, stride=1, padding=1, bias=False),
            # self.nonlinearity,
            # (BS, 1, 8, 8)
            nn.AvgPool2d(self.input_shape[0] // 16) # Average pooling per feature to do global average pooling
                             # One value in the kernel per pixel. This image is 8x8 here.
            # (BS, 1, 1, 1)
        )

    def forward(self, x, cond, given_z=None):
        # Encoding path
        enc = self.encoder(x)

        # Transformer
        aux, z = self.trans(enc, cond, given_z)
        # print(' -> (dec) aux', torch.max(aux), torch.min(aux))

        # Decoding path
        dec = torch.cat((aux, enc), dim=1)
        # print(' -> (dec) dec', torch.max(dec), torch.min(dec))

        output = self.judge(dec)

        return output, z


class ADNICondWGAN(CondWGAN):
    def __init__(self, params, attr_size, name="image_gan"):
        latent_dim = params["latent_dim"]
        # n_chan_enc = params["n_chan_enc"]
        # n_chan_gen = params["n_chan_gen"]
        finetune = params["finetune"]
        lr = params["lr"]
        d_updates_per_g_update = params["d_updates_per_g_update"]
        gradient_clip_val = params["gradient_clip_val"]

        args = {
            'latent_space_dim': latent_dim,
            'kernel_size': (3,3),
            'input_shape': (192, 192),
            'initial_filters': 32,
            'n_channels': 1,
            'use_sigmoid': False,
            'gen_params': {
                'norm': 'layernorm',
                'activation': 'relu'
            },
            'discr_params': {
                'norm': 'layernorm',
                'activation': 'relu'
            }
        }

        generator = GeneratorXia(args)
        discriminator = DiscriminatorXia(args)

        super().__init__(generator, discriminator, latent_dim, d_updates_per_g_update, gradient_clip_val, finetune, lr, name)
