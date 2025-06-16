import os
import numpy as np
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2

"""
推薦參數設定
super_batch_size = 2的倍數隨便
super_max_epoch = 20左右差不多 也可以多點
super_generator_model_type = 5 6 7 8 9都測測看
super_discriminator_model_type = 1 2 3都測測 2可能太強
super_patch_num = 32 16 8應該都不錯
super_use_yuv 比較用
super_time_stamp 就當名字 不要設定都一樣不然會被蓋過
"""



############################################超參數############################################
super_valid_and_tes_vs_train_ratio = 0.2 #若想要訓練:其他為8:2 則這邊設為0.2
super_test_vs_alid_ratio = 0.5 #若想要驗證:測試為1:1 則這邊設為0.5 整份的大小為上面的其他

super_batch_size = 2
super_max_epoch = 20
super_generator_model_type = "9" #版本幾 
super_discriminator_model_type = "3" #版本幾 
super_time_stamp = "20250603"
#只有辨別器type3有用 限定數值128 64 32 16 8 4 2 實際patch數量為super_patch_num^2
super_patch_num = 64
super_use_yuv = True


super_save_freq = 1 #每幾個epoch存檔
#別動
if super_discriminator_model_type == "3" and super_use_yuv:
    super_output_dir = f"./batch{super_batch_size}_epoch{super_max_epoch}_type{super_generator_model_type}{super_discriminator_model_type}_patchnum{super_patch_num}_yuv_{super_time_stamp}" #下次訓練如果還是用同樣的可能會覆蓋掉上次的
elif super_discriminator_model_type == "3" and super_use_yuv==False:
    super_output_dir = f"./batch{super_batch_size}_epoch{super_max_epoch}_type{super_generator_model_type}{super_discriminator_model_type}_patchnum{super_patch_num}_rgb_{super_time_stamp}" #下次訓練如果還是用同樣的可能會覆蓋掉上次的
elif super_use_yuv:
    super_output_dir = f"./batch{super_batch_size}_epoch{super_max_epoch}_type{super_generator_model_type}{super_discriminator_model_type}_yuv_{super_time_stamp}" #下次訓練如果還是用同樣的可能會覆蓋掉上次的
elif super_use_yuv==False:
    super_output_dir = f"./batch{super_batch_size}_epoch{super_max_epoch}_type{super_generator_model_type}{super_discriminator_model_type}_rgb_{super_time_stamp}" #下次訓練如果還是用同樣的可能會覆蓋掉上次的


############################################超參數############################################

"""
1.加入判別器accurary曲線 
2.新增生成器type5678和辨別器type3
3.新增可以YUV

生成器：
1.IN+NO_SHUFFLE -->type5
2.BN+NO_SHUFFLE -->type6

辨別器：
1.原版(可調patch size) -->type3
"""

#super_use_yuv = True則生出YUV圖片 需要改為RGB存檔
def yuv_tensor_to_rgb_tensor(yuv_tensor):
    # yuv_tensor: (3, H, W), float tensor, range [-1, 1]
    yuv = yuv_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    yuv = (yuv * 0.5 + 0.5) * 255.0  # scale to [0, 255]
    yuv = yuv.astype('uint8')

    # OpenCV expects YUV in specific format, this is YUV420/444, here we assume YUV444:
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgb = rgb.astype('float32') / 255.0  # normalize to [0,1]
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (3,H,W)

    return rgb_tensor.clamp(0, 1)

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立資料集類別
class MangaColorizationDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        bw_img = Image.fromarray(np.array(self.dataset[idx]['bw_image'])).convert('RGB')
        color_img = Image.fromarray(np.array(self.dataset[idx]['color_image'])).convert('RGB')
        if super_use_yuv:

            # 轉成 numpy (RGB)
            bw_np = np.array(bw_img)
            color_np = np.array(color_img)

            # 轉成 YUV
            bw_yuv = cv2.cvtColor(bw_np, cv2.COLOR_RGB2YUV)
            color_yuv = cv2.cvtColor(color_np, cv2.COLOR_RGB2YUV)

            # 再轉回 PIL (如果你的 transform 接受 PIL image)
            bw_yuv_img = Image.fromarray(bw_yuv)
            color_yuv_img = Image.fromarray(color_yuv)

            return self.transform(bw_yuv_img), self.transform(color_yuv_img)
        else:
            return self.transform(bw_img), self.transform(color_img)

######################################### 建立生成器 #########################################
#nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
class UNetGenerator(nn.Module): #這不是Unet
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), #縮兩倍
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), #縮兩倍
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), #縮兩倍
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class UNetGenerator_type2(nn.Module): #修改上下採樣，但也不是常見的shuffle用法
    def __init__(self):
        super(UNetGenerator_type2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1), 
            nn.PixelUnshuffle(2), # 通道變為8*2*2,尺寸變為H/2*W/2
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), 
            nn.PixelUnshuffle(2), # 通道變為32*2*2,尺寸變為H/2*W/2
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
            nn.PixelUnshuffle(2), # 通道變為128*2*2,尺寸變為H/2*W/2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), 
            nn.PixelShuffle(2), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), 
            nn.PixelShuffle(2), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=12, kernel_size=3, stride=1, padding=1), 
            nn.PixelShuffle(2), 
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#來自GPT的Unet
class ConvBlock(nn.Module):
    """基本卷積區塊：Conv + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """下採樣：PixelUnshuffle + Conv"""
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.down = nn.Sequential(
            nn.PixelUnshuffle(scale),
            nn.Conv2d(in_channels * (scale ** 2), out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """上採樣：Conv + PixelShuffle"""
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)
#來自GPT的Unet
class UNetGenerator_type3(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Encoder
        prev_channels = in_channels
        for feat in features:
            self.encoder_blocks.append(ConvBlock(prev_channels, feat))
            self.downsamples.append(Down(feat, feat))  # down after block
            prev_channels = feat

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2

        for feat in reversed_features:
            self.upsamples.append(Up(prev_channels, feat))
            self.decoder_blocks.append(ConvBlock(feat * 2, feat))  # skip + upsampled
            prev_channels = feat

        # Output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for enc, down in zip(self.encoder_blocks, self.downsamples):
            x = enc(x)
            skip_connections.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for up, dec, skip in zip(self.upsamples, self.decoder_blocks, skip_connections):
            x = up(x)
            x = torch.cat((x, skip), dim=1)
            x = dec(x)

        return self.final_conv(x)











#type3修改batch norm為 instance norm
class ConvBlock_IN(nn.Module):
    """基本卷積區塊：Conv + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Down_IN(nn.Module):
    """下採樣：PixelUnshuffle + Conv"""
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.down = nn.Sequential(
            nn.PixelUnshuffle(scale),
            nn.Conv2d(in_channels * (scale ** 2), out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)

class Up_IN(nn.Module):
    """上採樣：Conv + PixelShuffle"""
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)
#來自GPT的Unet
class UNetGenerator_type4(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Encoder
        prev_channels = in_channels
        for feat in features:
            self.encoder_blocks.append(ConvBlock_IN(prev_channels, feat))
            self.downsamples.append(Down_IN(feat, feat))  # down after block
            prev_channels = feat

        # Bottleneck
        self.bottleneck = ConvBlock_IN(features[-1], features[-1] * 2)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2

        for feat in reversed_features:
            self.upsamples.append(Up_IN(prev_channels, feat))
            self.decoder_blocks.append(ConvBlock_IN(feat * 2, feat))  # skip + upsampled
            prev_channels = feat

        # Output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for enc, down in zip(self.encoder_blocks, self.downsamples):
            x = enc(x)
            skip_connections.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for up, dec, skip in zip(self.upsamples, self.decoder_blocks, skip_connections):
            x = up(x)
            x = torch.cat((x, skip), dim=1)
            x = dec(x)

        return self.final_conv(x)



#幾乎依照論文描述實做 只是input output通道為3
class UNetGenerator_type5(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator_type5, self).__init__()

        # Encoder (Downsampling)
        self.down1 = nn.Sequential(                       # 256 -> 128
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(                       # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(                       # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down4 = nn.Sequential(                       # 32 -> 16
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down5 = nn.Sequential(                       # 16 -> 8
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down6 = nn.Sequential(                       # 8 -> 4
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder (Upsampling)
        self.up1 = nn.Sequential(                         # 4 -> 8
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(                         # 8 -> 16
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # skip
            nn.InstanceNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(                         # 16 -> 32
            nn.ConvTranspose2d(1024, 256, 4, 2, 1),  # skip
            nn.InstanceNorm2d(256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(                         # 32 -> 64
            nn.ConvTranspose2d(512, 128, 4, 2, 1),  # skip
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(                         # 64 -> 128
            nn.ConvTranspose2d(256, 64, 4, 2, 1),   # skip
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up6 = nn.Sequential(                         # 128 -> 256
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # skip
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up7 = nn.Sequential(                         # 128 -> 256
            nn.Conv2d(35, out_channels, 3, 1, 1),  # skip
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x) #64
        d2 = self.down2(d1) #128
        d3 = self.down3(d2) #256
        d4 = self.down4(d3) #512
        d5 = self.down5(d4) #512
        d6 = self.down6(d5) #512
        # print(f"d1:{d1.shape}")
        # print(f"d2:{d2.shape}")
        # print(f"d3:{d3.shape}")
        # print(f"d4:{d4.shape}")
        # print(f"d5:{d5.shape}")
        # print(f"d6:{d6.shape}")
        # print("============================")
        # Decoder + Skip connections
        u1 = self.up1(d6) 
        # print(f"u1:{u1.shape}")
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        # print(f"u2:{u2.shape} -->d5")
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        # print(f"u3:{u3.shape} -->d4")
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        # print(f"u4:{u4.shape} -->d3")
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        # print(f"u5:{u5.shape} -->d2")
        u6 = self.up6(torch.cat([u5, d1], dim=1))
        # print(f"u6:{u6.shape} -->d1")
        u7 = self.up7(torch.cat([u6, x], dim=1))
        # print(f"u7:{u7.shape} -->x")
        return u7


class UNetGenerator_type6(nn.Module): #type5 IN轉BN
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator_type6, self).__init__()

        # Encoder (Downsampling)
        self.down1 = nn.Sequential(                       # 256 -> 128
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(                       # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(                       # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down4 = nn.Sequential(                       # 32 -> 16
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down5 = nn.Sequential(                       # 16 -> 8
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down6 = nn.Sequential(                       # 8 -> 4
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder (Upsampling)
        self.up1 = nn.Sequential(                         # 4 -> 8
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(                         # 8 -> 16
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # skip
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(                         # 16 -> 32
            nn.ConvTranspose2d(1024, 256, 4, 2, 1),  # skip
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(                         # 32 -> 64
            nn.ConvTranspose2d(512, 128, 4, 2, 1),  # skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(                         # 64 -> 128
            nn.ConvTranspose2d(256, 64, 4, 2, 1),   # skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up6 = nn.Sequential(                         # 128 -> 256
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # skip
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up7 = nn.Sequential(                         # 128 -> 256
            nn.Conv2d(35, out_channels, 3, 1, 1),  # skip
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x) #64
        d2 = self.down2(d1) #128
        d3 = self.down3(d2) #256
        d4 = self.down4(d3) #512
        d5 = self.down5(d4) #512
        d6 = self.down6(d5) #512
        # print(f"d1:{d1.shape}")
        # print(f"d2:{d2.shape}")
        # print(f"d3:{d3.shape}")
        # print(f"d4:{d4.shape}")
        # print(f"d5:{d5.shape}")
        # print(f"d6:{d6.shape}")
        # print("============================")
        # Decoder + Skip connections
        u1 = self.up1(d6) 
        # print(f"u1:{u1.shape}")
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        # print(f"u2:{u2.shape} -->d5")
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        # print(f"u3:{u3.shape} -->d4")
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        # print(f"u4:{u4.shape} -->d3")
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        # print(f"u5:{u5.shape} -->d2")
        u6 = self.up6(torch.cat([u5, d1], dim=1))
        # print(f"u6:{u6.shape} -->d1")
        u7 = self.up7(torch.cat([u6, x], dim=1))
        # print(f"u7:{u7.shape} -->x")
        return u7

class UNetGenerator_type7(nn.Module): #type5 最後輸出層修改 因為type5輸出好像有點糊糊的
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator_type7, self).__init__()

        # Encoder (Downsampling)
        self.down1 = nn.Sequential(                       # 256 -> 128
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(                       # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(                       # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down4 = nn.Sequential(                       # 32 -> 16
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down5 = nn.Sequential(                       # 16 -> 8
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down6 = nn.Sequential(                       # 8 -> 4
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder (Upsampling)
        self.up1 = nn.Sequential(                         # 4 -> 8
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(                         # 8 -> 16
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # skip
            nn.InstanceNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(                         # 16 -> 32
            nn.ConvTranspose2d(1024, 256, 4, 2, 1),  # skip
            nn.InstanceNorm2d(256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(                         # 32 -> 64
            nn.ConvTranspose2d(512, 128, 4, 2, 1),  # skip
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(                         # 64 -> 128
            nn.ConvTranspose2d(256, 64, 4, 2, 1),   # skip
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up6 = nn.Sequential(                         # 128 -> 256
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # skip
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up7 = nn.Sequential(                         # 128 -> 256
            nn.Conv2d(32, out_channels, 3, 1, 1),  # skip
        )
        self.tanh = nn.Tanh()
    def forward(self, x):
        # Encoder
        d1 = self.down1(x) #64
        d2 = self.down2(d1) #128
        d3 = self.down3(d2) #256
        d4 = self.down4(d3) #512
        d5 = self.down5(d4) #512
        d6 = self.down6(d5) #512
        # print(f"d1:{d1.shape}")
        # print(f"d2:{d2.shape}")
        # print(f"d3:{d3.shape}")
        # print(f"d4:{d4.shape}")
        # print(f"d5:{d5.shape}")
        # print(f"d6:{d6.shape}")
        # print("============================")
        # Decoder + Skip connections
        u1 = self.up1(d6) 
        # print(f"u1:{u1.shape}")
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        # print(f"u2:{u2.shape} -->d5")
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        # print(f"u3:{u3.shape} -->d4")
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        # print(f"u4:{u4.shape} -->d3")
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        # print(f"u5:{u5.shape} -->d2")
        u6 = self.up6(torch.cat([u5, d1], dim=1))
        # print(f"u6:{u6.shape} -->d1")
        u7 = self.up7(u6)
        u8 = self.tanh(u7+x)
        
        # print(f"u7:{u7.shape} -->x")
        return u8


class UNetGenerator_type8(nn.Module): #type6 最後輸出層修改 因為type5輸出好像有點糊糊的
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator_type8, self).__init__()

        # Encoder (Downsampling)
        self.down1 = nn.Sequential(                       # 256 -> 128
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(                       # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(                       # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down4 = nn.Sequential(                       # 32 -> 16
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down5 = nn.Sequential(                       # 16 -> 8
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down6 = nn.Sequential(                       # 8 -> 4
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder (Upsampling)
        self.up1 = nn.Sequential(                         # 4 -> 8
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(                         # 8 -> 16
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # skip
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(                         # 16 -> 32
            nn.ConvTranspose2d(1024, 256, 4, 2, 1),  # skip
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(                         # 32 -> 64
            nn.ConvTranspose2d(512, 128, 4, 2, 1),  # skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(                         # 64 -> 128
            nn.ConvTranspose2d(256, 64, 4, 2, 1),   # skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up6 = nn.Sequential(                         # 128 -> 256
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # skip
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up7 = nn.Sequential(                         # 128 -> 256
            nn.Conv2d(32, out_channels, 3, 1, 1),  # skip
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        d1 = self.down1(x) #64
        d2 = self.down2(d1) #128
        d3 = self.down3(d2) #256
        d4 = self.down4(d3) #512
        d5 = self.down5(d4) #512
        d6 = self.down6(d5) #512
        # print(f"d1:{d1.shape}")
        # print(f"d2:{d2.shape}")
        # print(f"d3:{d3.shape}")
        # print(f"d4:{d4.shape}")
        # print(f"d5:{d5.shape}")
        # print(f"d6:{d6.shape}")
        # print("============================")
        # Decoder + Skip connections
        u1 = self.up1(d6) 
        # print(f"u1:{u1.shape}")
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        # print(f"u2:{u2.shape} -->d5")
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        # print(f"u3:{u3.shape} -->d4")
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        # print(f"u4:{u4.shape} -->d3")
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        # print(f"u5:{u5.shape} -->d2")
        u6 = self.up6(torch.cat([u5, d1], dim=1))
        # print(f"u6:{u6.shape} -->d1")
        u7 = self.up7(u6)
        u8 = self.tanh(u7+x)
        
        # print(f"u7:{u7.shape} -->x")
        return u8


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B*N*C/8
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B*C*N/8
        energy =  torch.bmm(proj_query,proj_key) # batch的matmul B*N*N
        attention = self.softmax(energy) # B * (N) * (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1, width*height) # B * C * N
 
        out = torch.bmm(proj_value,attention.permute(0,2,1) ) # B*C*N
        out = out.view(m_batchsize,C,width,height) # B*C*H*W
 
        out = self.gamma*out + x
        return out,attention


class UNetGenerator_type9(nn.Module): #type6 最後輸出層修改 因為type5輸出好像有點糊糊的
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator_type9, self).__init__()
        # Encoder (Downsampling)
        self.down1 = nn.Sequential(                       # 256 -> 128
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(                       # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(                       # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attention1 = Self_Attn(in_dim=256,activation=nn.LeakyReLU(0.2, inplace=True))
        self.down4 = nn.Sequential(                       # 32 -> 16
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down5 = nn.Sequential(                       # 16 -> 8
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down6 = nn.Sequential(                       # 8 -> 4
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attention2 = Self_Attn(in_dim=512,activation=nn.LeakyReLU(0.2, inplace=True))
        # Decoder (Upsampling)
        self.up1 = nn.Sequential(                         # 4 -> 8
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(                         # 8 -> 16
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # skip
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(                         # 16 -> 32
            nn.ConvTranspose2d(1024, 256, 4, 2, 1),  # skip
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(                         # 32 -> 64
            nn.ConvTranspose2d(512, 128, 4, 2, 1),  # skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(                         # 64 -> 128
            nn.ConvTranspose2d(256, 64, 4, 2, 1),   # skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up6 = nn.Sequential(                         # 128 -> 256
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # skip
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up7 = nn.Sequential(                         # 128 -> 256
            nn.Conv2d(32, out_channels, 3, 1, 1),  # skip
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        d1 = self.down1(x) #64
        d2 = self.down2(d1) #128
        d3 = self.down3(d2) #256
        d3,_ = self.attention1(d3)
        d4 = self.down4(d3) #512
        d5 = self.down5(d4) #512
        d6 = self.down6(d5) #512
        d6,_ = self.attention2(d6)
        # print(f"d1:{d1.shape}")
        # print(f"d2:{d2.shape}")
        # print(f"d3:{d3.shape}")
        # print(f"d4:{d4.shape}")
        # print(f"d5:{d5.shape}")
        # print(f"d6:{d6.shape}")
        # print("============================")
        # Decoder + Skip connections
        u1 = self.up1(d6) 
        # print(f"u1:{u1.shape}")
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        # print(f"u2:{u2.shape} -->d5")
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        # print(f"u3:{u3.shape} -->d4")
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        # print(f"u4:{u4.shape} -->d3")
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        # print(f"u5:{u5.shape} -->d2")
        u6 = self.up6(torch.cat([u5, d1], dim=1))
        # print(f"u6:{u6.shape} -->d1")
        u7 = self.up7(u6)
        u8 = self.tanh(u7+x)
        
        # print(f"u7:{u7.shape} -->x")
        return u8

######################################### 建立生成器 #########################################
######################################### 建立辨識器 #########################################
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1), #有點跳太多
            nn.Sigmoid()
        )

    def forward(self, input_A, input_B):
        x = torch.cat([input_A, input_B], dim=1)
        return self.model(x)
#來自GPT的resnet類似模組
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class Discriminator_type2(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 256, downsample=True),
            ResidualBlock(256, 512, downsample=True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()  # Optional: remove if using BCEWithLogitsLoss
        )

    def forward(self, input_A, input_B):
        x = torch.cat([input_A, input_B], dim=1)  # [B, 6, H, W]
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.final(x)
        return x


class Discriminator_type3(nn.Module):
    def __init__(self,super_patch_num):
        super(Discriminator_type3, self).__init__()
        if super_patch_num == 128: #patch_size 2*2
            self.model = nn.Sequential(
                nn.Conv2d(6, 64, 4, 2, 1), #128*128
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 1, 3, 1, 1), 
                nn.Sigmoid()
            )
        elif super_patch_num == 64: #patch_size 4*4
            self.model = nn.Sequential(
                nn.Conv2d(6, 64, 4, 2, 1), #128*128
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), #64*64
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 1, 3, 1, 1), 
                nn.Sigmoid()
            )
        elif super_patch_num == 32: #patch_size 8*8
            self.model = nn.Sequential(
                nn.Conv2d(6, 64, 4, 2, 1), #128*128
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), #64*64
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), #32*32
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 1, 3, 1, 1), 
                nn.Sigmoid()
            )
        elif super_patch_num == 16: #patch_size 16*16
            self.model = nn.Sequential(
                nn.Conv2d(6, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), #16*16
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 1, 3, 1, 1), 
                nn.Sigmoid()
            )
        elif super_patch_num == 8: #patch_size 32*32
            self.model = nn.Sequential(
                nn.Conv2d(6, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), #8*8
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 1, 3, 1, 1), 
                nn.Sigmoid()
            )
        elif super_patch_num == 4: #patch_size 64*64
            self.model = nn.Sequential(
                nn.Conv2d(6, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), #4*4
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 1, 3, 1, 1), 
                nn.Sigmoid()
            )
        elif super_patch_num == 2: #patch_size 64*64
            self.model = nn.Sequential(
                nn.Conv2d(6, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), 
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), #2*2
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 1, 3, 1, 1), 
                nn.Sigmoid()
            )
        

    def forward(self, input_A, input_B):
        x = torch.cat([input_A, input_B], dim=1)
        return self.model(x)
######################################### 建立辨識器 #########################################
# 訓練辨識器
def train_discriminator(real_A, real_B):
    optimizer_D.zero_grad()
    fake_B = netG(real_A).detach()
    pred_real = netD(real_A, real_B)
    pred_fake = netD(real_A, fake_B)
    loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
    loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    loss_D.backward()
    optimizer_D.step()
    # 假設 pred_real 和 pred_fake 是 shape: (B, 1, H, W)
    # 真圖：預測值 > 0.5 視為預測成功
    correct_real = (pred_real > 0.5).float().sum()
    total_real = pred_real.numel()
    acc_real = correct_real / total_real

    # 假圖：預測值 < 0.5 視為預測成功
    correct_fake = (pred_fake < 0.5).float().sum()
    total_fake = pred_fake.numel()
    acc_fake = correct_fake / total_fake

    return loss_D_real.item(), loss_D_fake.item(), loss_D.item(), acc_real.item(), acc_fake.item()

# 訓練生成器
def train_generator(real_A, real_B):
    optimizer_G.zero_grad()
    fake_B = netG(real_A)
    pred_fake = netD(real_A, fake_B)
    loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
    loss_L1 = criterion_L1(fake_B, real_B)
    loss_G = loss_GAN + 100 * loss_L1
    loss_G.backward()
    optimizer_G.step()
    return loss_GAN.item(), loss_L1.item()


# 測試辨識器(把訓練裡的倒傳遞和計算梯度拔掉)
def valid_discriminator(real_A, real_B):
    fake_B = netG(real_A).detach()
    pred_real = netD(real_A, real_B)
    pred_fake = netD(real_A, fake_B)
    loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
    loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
    loss_D = (loss_D_real + loss_D_fake) * 0.5

    # 假設 pred_real 和 pred_fake 是 shape: (B, 1, H, W)
    # 真圖：預測值 > 0.5 視為預測成功
    correct_real = (pred_real > 0.5).float().sum()
    total_real = pred_real.numel()
    acc_real = correct_real / total_real

    # 假圖：預測值 < 0.5 視為預測成功
    correct_fake = (pred_fake < 0.5).float().sum()
    total_fake = pred_fake.numel()
    acc_fake = correct_fake / total_fake

    return loss_D_real.item(), loss_D_fake.item(), loss_D.item(), acc_real.item(), acc_fake.item()

# 測試生成器(把訓練裡的倒傳遞和計算梯度拔掉)
def valid_generator(real_A, real_B):
    fake_B = netG(real_A)
    pred_fake = netD(real_A, fake_B)
    loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
    loss_L1 = criterion_L1(fake_B, real_B)
    loss_G = loss_GAN + 100 * loss_L1
    return loss_GAN.item(), loss_L1.item()


############################################資料集處理############################################
# 載入資料集

#export HF_HUB_CACHE=/home/dcmc/Data/xinjia/GAI/hub


dataset = load_dataset("MichaelP84/manga-colorization-dataset", split="train",
    cache_dir="/home/dcmc/Data/xinjia/GAI")

# 每次切出相同訓練、驗證、測試集
# 先拆出 20% 作為驗證+測試用（剩下 80% 當訓練集）
train_validtest = dataset.train_test_split(test_size=super_valid_and_tes_vs_train_ratio, seed=42)

# 再把 20% 拆成 10% + 10%（驗證與測試）
valid_test = train_validtest["test"].train_test_split(test_size=super_test_vs_alid_ratio, seed=42)

# 最終拆分
train_dataset = train_validtest["train"]      # 80%
valid_dataset = valid_test["train"]           # 10%
test_dataset  = valid_test["test"]            # 10%
############################################資料集處理############################################

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
valid_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
manga_dataset_train = MangaColorizationDataset(train_dataset, train_transform) #training dataset
train_dataloader = DataLoader(manga_dataset_train, batch_size=super_batch_size, shuffle=True)

manga_dataset_valid = MangaColorizationDataset(valid_dataset, valid_test_transform) #validation dataset
valid_dataloader = DataLoader(manga_dataset_valid, batch_size=super_batch_size, shuffle=False)

manga_dataset_test = MangaColorizationDataset(test_dataset, valid_test_transform) #testing dataset
test_dataloader = DataLoader(manga_dataset_test, batch_size=super_batch_size, shuffle=False)

# 初始化網路與損失函數
if super_generator_model_type == "1":
    netG = UNetGenerator().to(device)
elif super_generator_model_type == "2":
    netG = UNetGenerator_type2().to(device)
elif super_generator_model_type == "3":
    netG = UNetGenerator_type3().to(device)
elif super_generator_model_type == "4":
    netG = UNetGenerator_type4().to(device)
elif super_generator_model_type == "5":
    netG = UNetGenerator_type5().to(device)
elif super_generator_model_type == "6":
    netG = UNetGenerator_type6().to(device)
elif super_generator_model_type == "7":
    netG = UNetGenerator_type7().to(device)
elif super_generator_model_type == "8":
    netG = UNetGenerator_type8().to(device)
elif super_generator_model_type == "9":
    netG = UNetGenerator_type9().to(device)
else:
    netG = UNetGenerator().to(device)
if super_discriminator_model_type == "1":
    netD = Discriminator().to(device)
elif super_discriminator_model_type == "2":
    netD = Discriminator_type2().to(device)
else:
    netD = Discriminator().to(device)

criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()
optimizer_G = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 訓練超參數設置
max_epochs = super_max_epoch
sample_interval = super_save_freq
output_dir = super_output_dir
os.makedirs(output_dir, exist_ok=True)

# 儲存loss的list
all_loss_D_real_train = []
all_loss_D_fake_train = []
all_loss_D_total_train = []
all_loss_G_GAN_train = []
all_loss_G_L1_train = []
all_loss_G_total_train = []
all_predict_real_accuracy_train = []
all_predict_fake_accuracy_train = []


all_loss_D_real_valid = []
all_loss_D_fake_valid = []
all_loss_D_total_valid = []
all_loss_G_GAN_valid = []
all_loss_G_L1_valid = []
all_loss_G_total_valid = []
all_predict_real_accuracy_valid = []
all_predict_fake_accuracy_valid = []

# 訓練
for epoch in range(max_epochs):
    epoch_loss_D_real_train = 0.0
    epoch_loss_D_fake_train = 0.0
    epoch_loss_D_total_train = 0.0
    epoch_loss_G_GAN_train = 0.0
    epoch_loss_G_L1_train = 0.0
    epoch_loss_G_total_train = 0.0


    epoch_acc_D_real_train = 0.0
    epoch_acc_D_fake_train = 0.0
    netG.train()
    netD.train()
    
    num_steps = len(train_dataloader)
    for i, (real_A, real_B) in enumerate(train_dataloader):
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        loss_D_real, loss_D_fake, loss_D,acc_real,acc_fake = train_discriminator(real_A, real_B)
        loss_GAN, loss_L1 = train_generator(real_A, real_B)

        #紀錄loss
        epoch_loss_D_real_train += loss_D_real
        epoch_loss_D_fake_train += loss_D_fake
        epoch_loss_D_total_train += loss_D
        epoch_loss_G_GAN_train += loss_GAN
        epoch_loss_G_L1_train += loss_L1
        epoch_loss_G_total_train += loss_GAN + 100 * loss_L1

        epoch_acc_D_real_train += acc_real
        epoch_acc_D_fake_train += acc_fake
        print(f"[Epoch {epoch+1}/{max_epochs}] [Batch {i+1}/{len(train_dataloader)}] "
              f"D_real: {loss_D_real:.4f}, D_fake: {loss_D_fake:.4f}, D_total: {loss_D:.4f} "
              f"G_GAN: {loss_GAN:.4f}, G_L1: {loss_L1:.4f}, G_total: {loss_GAN + 100 * loss_L1}")
    #紀錄loss
    all_loss_D_real_train.append(epoch_loss_D_real_train/num_steps)
    all_loss_D_fake_train.append(epoch_loss_D_fake_train/num_steps)
    all_loss_D_total_train.append(epoch_loss_D_total_train/num_steps)
    all_loss_G_GAN_train.append(epoch_loss_G_GAN_train/num_steps)
    all_loss_G_L1_train.append(epoch_loss_G_L1_train/num_steps)
    all_loss_G_total_train.append(epoch_loss_G_total_train/num_steps)
    all_predict_real_accuracy_train.append(epoch_acc_D_real_train/num_steps)
    all_predict_fake_accuracy_train.append(epoch_acc_D_fake_train/num_steps)
    print("=============================================================================================")
    print(f"[Epoch {epoch+1}] training: "
              f"D_real: {epoch_loss_D_real_train/num_steps:.4f}, D_fake: {epoch_loss_D_fake_train/num_steps:.4f}, D_total: {epoch_loss_D_total_train/num_steps:.4f} "
              f"G_GAN: {epoch_loss_G_GAN_train/num_steps:.4f}, G_L1: {epoch_loss_G_L1_train/num_steps:.4f}, G_total: {epoch_loss_G_total_train/num_steps}, D_predict_real: {epoch_acc_D_real_train/num_steps}, D_predict_fake: {epoch_acc_D_fake_train/num_steps}")
    print("=============================================================================================")
    
    
    
    epoch_loss_D_real_valid = 0.0
    epoch_loss_D_fake_valid = 0.0
    epoch_loss_D_total_valid = 0.0
    epoch_loss_G_GAN_valid = 0.0
    epoch_loss_G_L1_valid = 0.0
    epoch_loss_G_total_valid = 0.0

    epoch_acc_D_real_valid = 0.0
    epoch_acc_D_fake_valid = 0.0
    netG.eval()
    netD.eval()
    

    num_steps = len(valid_dataloader)
    for i, (real_A, real_B) in enumerate(valid_dataloader):
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        loss_D_real, loss_D_fake, loss_D,acc_real,acc_fake  = valid_discriminator(real_A, real_B)
        loss_GAN, loss_L1 = valid_generator(real_A, real_B)

        #紀錄loss
        epoch_loss_D_real_valid += loss_D_real
        epoch_loss_D_fake_valid += loss_D_fake
        epoch_loss_D_total_valid += loss_D
        epoch_loss_G_GAN_valid += loss_GAN
        epoch_loss_G_L1_valid += loss_L1
        epoch_loss_G_total_valid += loss_GAN + 100 * loss_L1

        epoch_acc_D_real_valid += acc_real
        epoch_acc_D_fake_valid += acc_fake
        print(f"[Epoch {epoch+1}/{max_epochs}] [Batch {i+1}/{len(valid_dataloader)}] "
              f"D_real: {loss_D_real:.4f}, D_fake: {loss_D_fake:.4f}, D_total: {loss_D:.4f} "
              f"G_GAN: {loss_GAN:.4f}, G_L1: {loss_L1:.4f}, G_total: {loss_GAN + 100 * loss_L1}")
        
    #紀錄loss
    all_loss_D_real_valid.append(epoch_loss_D_real_valid/num_steps)
    all_loss_D_fake_valid.append(epoch_loss_D_fake_valid/num_steps)
    all_loss_D_total_valid.append(epoch_loss_D_total_valid/num_steps)
    all_loss_G_GAN_valid.append(epoch_loss_G_GAN_valid/num_steps)
    all_loss_G_L1_valid.append(epoch_loss_G_L1_valid/num_steps)
    all_loss_G_total_valid.append(epoch_loss_G_total_valid/num_steps)
    all_predict_real_accuracy_valid.append(epoch_acc_D_real_valid/num_steps)
    all_predict_fake_accuracy_valid.append(epoch_acc_D_fake_valid/num_steps)
    print("=============================================================================================")
    print(f"[Epoch {epoch+1}] validation: "
              f"D_real: {epoch_loss_D_real_valid/num_steps:.4f}, D_fake: {epoch_loss_D_fake_valid/num_steps:.4f}, D_total: {epoch_loss_D_total_valid/num_steps:.4f} "
              f"G_GAN: {epoch_loss_G_GAN_valid/num_steps:.4f}, G_L1: {epoch_loss_G_L1_valid/num_steps:.4f}, G_total: {epoch_loss_G_total_train/num_steps}, D_predict_real: {epoch_acc_D_real_valid/num_steps}, D_predict_fake: {epoch_acc_D_fake_valid/num_steps}")
    print("=============================================================================================")
    

    if (epoch + 1) % sample_interval == 0 or (epoch + 1) == max_epochs: #每sample_interval個或最後一個存
        with torch.no_grad():
            fake_B = netG(real_A)
        result = torch.cat([real_A[0], real_B[0], fake_B[0]], dim=2)
        if super_use_yuv:
            result = yuv_tensor_to_rgb_tensor(result)
        else:
            result = (result * 0.5 + 0.5).clamp(0, 1)
        save_image(result, os.path.join(output_dir, f"epoch_{epoch+1:04d}.png"))
        torch.save(netG.state_dict(), os.path.join(output_dir, f"netG_epoch_{epoch+1:04d}.pth"))
        torch.save(netD.state_dict(), os.path.join(output_dir, f"netD_epoch_{epoch+1:04d}.pth"))

print("==========================================training==========================================")
print(f"all_loss_D_real_train:{all_loss_D_real_train}")
print(f"all_loss_D_fake_train:{all_loss_D_fake_train}")
print(f"all_loss_D_total_train:{all_loss_D_total_train}")
print(f"all_loss_G_GAN_train:{all_loss_G_GAN_train}")
print(f"all_loss_G_L1_train:{all_loss_G_L1_train}")
print(f"all_loss_G_total_train:{all_loss_G_total_train}")
print(f"all_predict_real_accuracy_train:{all_predict_real_accuracy_train}")
print(f"all_predict_fake_accuracy_train:{all_predict_fake_accuracy_train}")
print("==========================================training==========================================")
print("==========================================validation==========================================")
print(f"all_loss_D_real_valid:{all_loss_D_real_valid}")
print(f"all_loss_D_fake_valid:{all_loss_D_fake_valid}")
print(f"all_loss_D_total_valid:{all_loss_D_total_valid}")
print(f"all_loss_G_GAN_valid:{all_loss_G_GAN_valid}")
print(f"all_loss_G_L1_valid:{all_loss_G_L1_valid}")
print(f"all_loss_G_total_valid:{all_loss_G_total_valid}")
print(f"all_predict_real_accuracy_valid:{all_predict_real_accuracy_valid}")
print(f"all_predict_fake_accuracy_valid:{all_predict_fake_accuracy_valid}")

print("==========================================validation==========================================")

# --- 儲存 loss list 成 txt ---
loss_dict = {
    "all_loss_D_real_train": all_loss_D_real_train,
    "all_loss_D_fake_train": all_loss_D_fake_train,
    "all_loss_D_total_train": all_loss_D_total_train,
    "all_loss_G_GAN_train": all_loss_G_GAN_train,
    "all_loss_G_L1_train": all_loss_G_L1_train,
    "all_loss_G_total_train": all_loss_G_total_train,
    "all_loss_D_real_valid": all_loss_D_real_valid,
    "all_loss_D_fake_valid": all_loss_D_fake_valid,
    "all_loss_D_total_valid": all_loss_D_total_valid,
    "all_loss_G_GAN_valid": all_loss_G_GAN_valid,
    "all_loss_G_L1_valid": all_loss_G_L1_valid,
    "all_loss_G_total_valid": all_loss_G_total_valid,
    "all_predict_real_accuracy_train": all_predict_real_accuracy_train,
    "all_predict_real_accuracy_valid": all_predict_real_accuracy_valid,
    "all_predict_fake_accuracy_train": all_predict_fake_accuracy_train,
    "all_predict_fake_accuracy_valid": all_predict_fake_accuracy_valid,
}

for name, loss_list in loss_dict.items():
    file_path = os.path.join(output_dir, f"{name}.txt")
    with open(file_path, "w") as f:
        for value in loss_list:
            f.write(f"{value:.6f}\n")  # 儲存小數點六位

#testing
epoch_loss_D_real_test = 0.0
epoch_loss_D_fake_test = 0.0
epoch_loss_D_total_test = 0.0
epoch_loss_G_GAN_test = 0.0
epoch_loss_G_L1_test = 0.0
epoch_loss_G_total_test = 0.0

epoch_acc_D_real_test = 0.0
epoch_acc_D_fake_test = 0.0
netG.eval()
netD.eval()
    

num_steps = len(test_dataloader)
for i, (real_A, real_B) in enumerate(test_dataloader):
    real_A = real_A.to(device)
    real_B = real_B.to(device)
    loss_D_real, loss_D_fake, loss_D,acc_real,acc_fake = valid_discriminator(real_A, real_B)
    loss_GAN, loss_L1 = valid_generator(real_A, real_B)

    #紀錄loss
    epoch_loss_D_real_test += loss_D_real
    epoch_loss_D_fake_test += loss_D_fake
    epoch_loss_D_total_test += loss_D
    epoch_loss_G_GAN_test += loss_GAN
    epoch_loss_G_L1_test += loss_L1
    epoch_loss_G_total_test += loss_GAN + 100 * loss_L1
    epoch_acc_D_real_test += acc_real
    epoch_acc_D_fake_test += acc_fake
    print(f"[Testing] [Batch {i+1}/{len(test_dataloader)}] "
          f"D_real: {loss_D_real:.4f}, D_fake: {loss_D_fake:.4f}, D_total: {loss_D:.4f} "
          f"G_GAN: {loss_GAN:.4f}, G_L1: {loss_L1:.4f}, G_total: {loss_GAN + 100 * loss_L1}")
print("=============================================================================================")
print(f"testing: "
          f"D_real: {epoch_loss_D_real_test/num_steps:.4f}, D_fake: {epoch_loss_D_fake_test/num_steps:.4f}, D_total: {epoch_loss_D_total_test/num_steps:.4f} "
          f"G_GAN: {epoch_loss_G_GAN_test/num_steps:.4f}, G_L1: {epoch_loss_G_L1_test/num_steps:.4f}, G_total: {epoch_loss_G_total_test/num_steps}, D_predict_real: {epoch_acc_D_real_test/num_steps}, D_predict_fake: {epoch_acc_D_fake_test/num_steps}")
print("=============================================================================================")







##畫出來(只有total loss有，剩下去看txt)
epochs = list(range(1, max_epochs + 1))  # x 軸：1 到 max_epochs

# 🎨 圖 1：Discriminator Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, all_loss_D_total_train, label='Train Loss D', color='blue')
plt.plot(epochs, all_loss_D_total_valid, label='Valid Loss D', color='orange')
plt.title("Discriminator Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(output_dir,"loss_discriminator.png")
plt.savefig(save_path)  # 儲存成圖片
plt.show()

# 🎨 圖 2：Generator Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, all_loss_G_total_train, label='Train Loss G', color='green')
plt.plot(epochs, all_loss_G_total_valid, label='Valid Loss G', color='red')
plt.title("Generator Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(output_dir,"loss_generator.png")
plt.savefig(save_path)  # 儲存成圖片
plt.show()

# 🎨 圖 3：predict acc
plt.figure(figsize=(10, 5))
plt.plot(epochs, all_predict_real_accuracy_train, label='Train predict real')
plt.plot(epochs, all_predict_real_accuracy_valid, label='Valid predict real')
plt.plot(epochs, all_predict_fake_accuracy_train, label='Train predict fake')
plt.plot(epochs, all_predict_fake_accuracy_valid, label='Valid predict fake')
plt.title("Discriminator predict acc Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("acc")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(output_dir,"Discriminator_predict_acc.png")
plt.savefig(save_path)  # 儲存成圖片
plt.show()

