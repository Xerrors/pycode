import torch.nn as nn

from utils.tricks import weights_init

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_gen_feature=64, num_channel=3):
        """
        Generator Code
        :param z_dim:            Size of z latent vector (i.e. size of generator input)
        :param num_gen_feature:  Size of feature maps in generator
        :param num_channel:      Number of channels in the training images. For color images this is 3
        """
        super(Generator, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( z_dim, num_gen_feature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_gen_feature * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(num_gen_feature * 8, num_gen_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_feature * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(num_gen_feature * 4, num_gen_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_feature * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(num_gen_feature * 2, num_gen_feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_feature),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(num_gen_feature, num_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, num_dis_feature=64, num_channel=3):
        """
        Discriminator Code
        :param num_dis_feature: Size of feature maps in discriminator
        :param num_channel:     Number of channels in the training images. For color images this is 3
        """
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_channel, num_dis_feature, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_dis_feature, num_dis_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_dis_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_dis_feature * 2, num_dis_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_dis_feature * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_dis_feature * 4, num_dis_feature * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_dis_feature * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_dis_feature * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def DCGAN(device, z_dim=100, num_gen_feature=64, num_dis_feature=64, num_channel=3):
    G = Generator(z_dim, num_gen_feature, num_channel).to(device)
    D = Discriminator(num_dis_feature, num_channel).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    return G, D