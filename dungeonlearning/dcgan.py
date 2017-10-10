import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as tdata

class DCGAN_D(nn.Module):
    # input channels (1), output channels, kernel_size, stride, padding
    def conv_block(self, main, name, inf, of, a, b, c, bn=True):
        convName = str.format("{}-{}.{}.conv", name, inf, of)
        main.add_module(convName, nn.Conv2d(inf, of, a, b, c, bias=False))
        batchName = str.format("{}-{}.batchnorm", name, of)
        main.add_module(batchName, nn.BatchNorm2d(of))
        reluName = str.format("{}-{}.relu", name, of)
        main.add_module(reluName, nn.LeakyReLU(0.2, inplace=True))

        #netD = DCGAN_D(sz, 1, 64, 1, 1).cuda()
    def __init__(self, isize, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        self.conv_block(main, 'initial', nc, ndf, 4, 2, 1, False)
        csize, cndf = isize / 2, ndf

        for t in range(n_extra_layers):
            extraName = str.format("extra-{}", t)
            # input and output of ndf (64 channels), size 3 kernel
            # no decrease in image size
            self.conv_block(main, extraName, cndf, cndf, 3, 1, 1)

        while csize > 4:
            # 4-size kernel, 2 stride, reduces size by factor of 2 (apparently)
            # continue until size is 4 (must have been multiple of 16)
            self.conv_block(main, 'pyramid', cndf, cndf*2, 4, 2, 1)
            cndf *= 2; csize /= 2

        # state size. K x 4 x 4
        cndfName = str.format("final.{}-1.conv", cndf)
        # Output to single channel, 4x4 for 0/1 discrimination
        # mean occurs in output
        main.add_module(cndfName, nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        output = output.mean(0)
        return output.view(1)

class DCGAN_G(nn.Module):
    def deconv_block(self, main, name, inf, of, a, b, c, bn=True):
        convtName = str.format("{}-{}.{}.convt", name, inf, of)
        main.add_module(convtName, nn.ConvTranspose2d(inf, of, a, b, c, bias=False))
        batchnormName = str.format("{}-{}.batchnorm", name, of)
        main.add_module(batchnormName, nn.BatchNorm2d(of))
        reluName = str.format("{}-{}.relu", name, of)
        main.add_module(reluName, nn.ReLU(inplace=True))

        #netG = DCGAN_G(sz, nz, 1, 64, 1, 1).cuda()
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        #cngf = 32 * 16 (sz / 4) = 512
        cngf, tisize = ngf//2, 4
        while tisize != isize: cngf *= 2; tisize *= 2

        main = nn.Sequential()
        # kernel size 4 deconv on original noise
        self.deconv_block(main, 'initial', nz, cngf, 4, 1, 0)

        csize, cndf = 4, cngf
        while csize < isize//2:
            # Reduce filters by half, double size of image, until half original size
            # Perform 5 times
            self.deconv_block(main, 'pyramid', cngf, cngf//2, 4, 2, 1)
            cngf //= 2; csize *= 2

        for t in range(n_extra_layers):
            extraName = str.format("extra-{}", t)
            # Extra deconv, does not affect size
            self.deconv_block(main, extraName, cngf, cngf, 3, 1, 1)

        cnfgName = str.format("final.{}-{}.convt", cngf, nc)
        # Final deconv to full sized image
        main.add_module(cnfgName, nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        ncName = str.format("final.{}.sigmoid", nc)
        main.add_module(ncName, nn.Sigmoid())
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)

    # Attempt to use upsampling in G rather than deconv. May suffer from limited number of channels.
class DCGAN_GU(nn.Module):
    def upsample_block(self, main, name, inf, of, a, b, c, bn=True):
        convtName = str.format("{}-{}.{}.convt", name, inf, of)
        #main.add_module(convtName, nn.ConvTranspose2d(inf, of, a, b, c, bias=False))
        main.add_module(convtName, nn.Upsample(scale_factor=2, mode="nearest"))
        batchnormName = str.format("{}-{}.batchnorm", name, of)
        main.add_module(batchnormName, nn.BatchNorm2d(of))
        reluName = str.format("{}-{}.relu", name, of)
        main.add_module(reluName, nn.ReLU(inplace=True))

    def deconv_block(self, main, name, inf, of, a, b, c, bn=True):
        convtName = str.format("{}-{}.{}.convt", name, inf, of)
        main.add_module(convtName, nn.ConvTranspose2d(inf, of, a, b, c, bias=False))
        #main.add_module(convtName, nn.Upsample(scale_factor=2, mode="nearest"))
        batchnormName = str.format("{}-{}.batchnorm", name, of)
        main.add_module(batchnormName, nn.BatchNorm2d(of))
        reluName = str.format("{}-{}.relu", name, of)
        main.add_module(reluName, nn.ReLU(inplace=True))

        #netG = DCGAN_G(sz, nz, 1, 64, 1, 1).cuda()
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_GU, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        #cngf = 32 * 16 (sz / 4) = 512
#        cngf, tisize = ngf//2, 4
#        while tisize != isize: cngf *= 2; tisize *= 2
        cngf = 64
                        
        main = nn.Sequential()
        # kernel size 4 deconv on original noise
        self.deconv_block(main, 'initial', nz, cngf, 4, 1, 0)

        csize, cndf = 4, cngf
        while csize < isize//2:
            # Reduce filters by half, double size of image, until half original size
            # Perform 5 times
            self.upsample_block(main, str.format('pyramid-{}',csize), cngf, cngf, 4, 2, 1)
            csize *= 2

        #for t in range(n_extra_layers):
        #    extraName = str.format("extra-{}", t)
        #    # Extra deconv, does not affect size
        #    self.deconv_block(main, extraName, cngf, cngf, 3, 1, 1)

        cnfgName = str.format("final.{}-{}.convt", cngf, nc)
        # Final deconv to full sized image
        #main.add_module(cnfgName, nn.Upsample(scale_factor=2, mode="nearest"))
        main.add_module(cnfgName, nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))

        ncName = str.format("final.{}.sigmoid", nc)
        main.add_module(ncName, nn.Sigmoid())
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)
