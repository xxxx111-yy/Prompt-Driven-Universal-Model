#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR,Logger,ReplayBuffer
from .utils import weights_init_normal,get_config,Normalize
from .datasets import ImageDataset_prompt,ValDataset_prompt
from Model.networks import *
from .utils import Resize,ToTensor
from .utils import Logger
from torchvision.transforms import RandomAffine
from torchvision.transforms import RandomAffine,ToPILImage
from .transformer import Transformer_2D
from skimage import measure

from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision
import torch
import imageio
from PIL import Image
from .networks1 import define_F,PatchSampleF
from .patchnce import PatchNCELoss


    
class CycPrompt_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.netG_A2B = GeneratorPromptEnhanced(config['input_nc'], config['output_nc']).cuda()
        self.vgg_loss = VGGLoss(1)
        self.netF = PatchSampleF(use_mlp=True, init_type='normal', init_gain=0.02, gpu_ids=[1], nc=256).cuda()
        self.netD_B = Discriminator(self.config["input_nc"]).cuda()  # discriminator for domain b
	
        params = list(self.netF.parameters())
        if len(params) == 0:
            print("Warning: netF has no parameters!")
        else:
            print(f"Number of parameters in netF: {len(params)}")
            self.optimizer_F = torch.optim.Adam(params, lr=0.0002, betas=(0.5, 0.999))
    
    
        #self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=0.0002,betas=(0.5, 0.999))
        self.nce_layers = '0,1,2,3,4,5'
        self.nce_layers = [int(i) for i in self.nce_layers.split(',')]
        self.criterionNCE = []

        for nce_layer in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss().cuda())
                
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999),weight_decay=0.0001)
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        self.BCE_loss = torch.nn.BCEWithLogitsLoss()
        
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.input_A_label = Tensor([1]).long()
        self.input_B_label = Tensor([1]).long()
        

        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
        
        def __make_power_2(img, base, method=Image.BICUBIC):
            ow, oh = img.size
            h = int(round(oh / base) * base)
            w = int(round(ow / base) * base)
            if h == oh and w == ow:
                return img

            return img.resize((w, h), method)
        
        #Dataset loader
        transforms_1 = [
                        transforms.Resize([1024,1024], Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        #Resize(size_tuple = (config['size'], config['size']))
                        ]
    
    
        self.dataloader = DataLoader(ImageDataset_MIST_prompt(config['dataroot'], transforms_1=transforms_1, transforms_2=transforms_1,  transforms_3=transforms_1, unaligned=False),
                                batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])

        val_transforms = [ToTensor(),
                          Resize(size_tuple = (config['size'], config['size']))]
        
        self.val_data = DataLoader(ValDataset_MIST_prompt(config['val_dataroot'],  transforms_1=transforms_1, transforms_2=transforms_1,  transforms_3=transforms_1, unaligned=False),
                                batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])


       # Loss plot
        self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))       
        
    def train(self):
        ###### Training ######
        #self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + '24_netG_A2B.pth'))

            
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            
            for i, batch in enumerate(self.dataloader):
                
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                
                A_label = Variable(self.input_A_label.copy_(batch['A_label']))
                B_label = Variable(self.input_A_label.copy_(batch['B_label']))

                self.optimizer_G.zero_grad()
                self.optimizer_F.zero_grad()
                # GAN loss

                fake_B2A = self.netG_A2B(real_B,A_label)
                fake_B2A2B = self.netG_A2B(fake_B2A,B_label)
                fake_B2B = self.netG_A2B(real_B,B_label) #identity
                #print('11111',real_A.size(0))
                pred_fake = self.netD_B(fake_B2A)
                loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                G_recon_loss_A = self.L1_loss(fake_B2A2B, real_B)

                G_identity_loss_A = self.L1_loss(fake_B2B, real_B)
                self.loss_NCE1 = self.calculate_NCE_loss(real_B, fake_B2A,A_label,A_label)
                # Total loss
                loss_Total = loss_GAN_A2B + G_recon_loss_A + G_identity_loss_A  + self.loss_NCE1
                loss_Total.backward()
                
                self.optimizer_F.step()
                self.optimizer_G.step()
                        

                ###### Discriminator B ######
                self.optimizer_D_B.zero_grad()

                # Real loss
                pred_real = self.netD_B(real_A)
                loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                # Fake loss
                fake_B2A = self.fake_B_buffer.push_and_pop(fake_B2A)
                pred_fake = self.netD_B(fake_B2A.detach())
                loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)


                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)
                loss_D_B.backward()

                self.optimizer_D_B.step()
                
  
                self.logger.log({
                                'loss_cycle_BAB':loss_Total,
                                'G_recon_loss_A':G_recon_loss_A,
                                'loss_D_A':loss_D_B,
                                 },
                                images={'rd_real_B': real_B, 'rd_real_A': real_A, 'fake_B2A':fake_B2A})
            if epoch>=50 or epoch%5==0:
            	torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + '%s_netG_A2B.pth'%epoch)

            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')
                    
                    
    def calculate_NCE_loss(self, src, tgt,src_ph, tgt_ph):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_A2B(tgt,tgt_ph, self.nce_layers,encode_only=True)

        if False and True and (np.random.random() < 0.5):
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG_A2B(src,src_ph, self.nce_layers,encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, 256, None)
        feat_q_pool, _ = self.netF(feat_q, 256, sample_ids)
        
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):

            loss = crit(f_q.squeeze(0), f_k.squeeze(0)) * 1
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


    

    

