import math
import ray
from ray import tune
from ray.air import session
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset,Dataset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
import cv2
from skimage.util import random_noise
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

seed = 4912
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, gauss_noise=True, gauss_blur=True, resize=128, p=0.5, center_crop=True):
        self.p = p
        self.resize = resize
        self.gauss_noise = gauss_noise
        self.gauss_blur = gauss_blur
        self.image_paths = image_paths
        self.center_crop = center_crop

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        gt_image = plt.imread(self.image_paths[idx])
        gt_image = self.__crop_center(gt_image)

        image = np.copy(gt_image)
        image = self.__apply_resize(image)
        image = self.__apply_gauss_blur(image)
        image = self.__apply_gauss_noise(image)
        image = np.transpose(image)
        gt_image = np.transpose(gt_image)
        
        image = image/255
        gt_image = gt_image/255
        return image.astype('float32'), gt_image.astype('float32') # so it matches the bias when train

    def __crop_center(self, image:np.array):
        h, w, _ = image.shape
        if h > w:
            start = (h - w) // 2
            image = image[start:start + w, :, :]
        elif w > h:    
            start = (w - h) // 2
            image = image[:, start:start + h, :]

        return cv2.resize(image, dsize=[self.resize] * 2, interpolation=cv2.INTER_CUBIC)
        
    def __apply_gauss_noise(self, image:np.array):
        if self.gauss_noise:
            for y in range(self.resize):
                for x in range(self.resize):
                    if random.randint(0, 2) / 2 <= self.p:
                        col = image[y][x]
                        noise = random.randint(-50, 50)
                        new_col = col + noise
                        image[y][x] = np.uint8(new_col)

        return image
    
    def __apply_gauss_blur(self, image:np.array):
        if random.randint(0, 2) / 2 <= self.p:
            cv2.GaussianBlur(image, 
                            [random.randint(1, 5) * 2 + 1] * 2,
                            0)

        return image
    
    def __apply_resize(self, image:np.array):
        return cv2.resize(image, dsize=[self.resize] * 2, interpolation=cv2.INTER_CUBIC)

class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DownSamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpSamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self, channels=[64, 128, 256], input_channels=3, output_channels=3):
        super().__init__()
        use_channels = channels + channels[::-1][1:]
        self.layer = nn.ModuleList([])
        self.layer.append(nn.Conv2d(in_channels=input_channels, out_channels=channels[0], kernel_size=3, stride=1, padding=1))
        for i in range(len(use_channels) - 1):
            channel_1 = use_channels[i]
            channel_2 = use_channels[i + 1]
            if i < len(channels) - 1:
                layer = DownSamplingBlock(channel_1, channel_2, kernel_size=3, stride=1, padding=1)
            else:
                layer = UpSamplingBlock(channel_1, channel_2, kernel_size=3, stride=1, padding=1)
            self.layer.append(layer)
                
        self.layer.append(
            nn.Conv2d(in_channels=channels[0], out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        for e in self.layer:
            x = e(x)
        
        return x

def save_model(model, dir, epoch, last=False):
    cur_time = datetime.datetime.now()
    file_name = f"model-at-{cur_time.strftime('%d-%m-%y-%H%M%S')}-epoch-{epoch}.pth"
    save_path = os.path.join(dir, file_name)

    try:
        if not os.path.exists(dir):
            os.makedirs(dir)

        if last:
            torch.save(model.state_dict(), os.path.join(dir, "last_model.pth"))
            print(f"Last model saved to {os.path.join(dir, 'last_model.pth')}")
        else:
            torch.save(model.state_dict(), save_path)
            torch.save(model, save_path)
            print(f"Model saved to {save_path}")

    except OSError as e:
        print(f"OS error while saving model: {e}")
    except Exception as e:
        print(f"Failed to save model due to unexpected error: {e}")


def train(model,opt,loss_fn,train_loader,test_loader,epochs=10,save_dir=None,device='cpu', report=None):
    print("🤖Training on", device)
    model = model.to(device)
    for epoch in range(epochs):
        running_loss = 0
        test_loss = 0
        ssim_val = 0
        psnr_val = 0

        model.train(True)
        train_bar = tqdm(train_loader,desc=f'🚀Training Epoch [{epoch+1}/{epochs}]',unit='batch')
        for images, gt in train_bar:
            opt.zero_grad()

            images = images.to(device)
            gt = gt.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, gt)
            loss.backward()

            opt.step()
            running_loss += loss.item()/epochs
            train_bar.set_postfix(running_loss=loss.item())
            
        model.eval()

        test_bar = tqdm(test_loader,desc='📄Testing',unit='batch')
        with torch.no_grad():
            for images, gt in test_bar:
                images = images.to(device)
                gt = gt.to(device)

                outputs = model(images)

                loss = loss_fn(outputs, gt)
                test_loss += loss.item()/epochs
                
                images_np = images.cpu().detach().numpy()
                gt_np = gt.cpu().detach().numpy()
                
                ssim_val += ssim(gt_np, images_np, data_range=1.0, channel_axis=1)/epochs
                psnr_val += psnr(gt_np, images_np, data_range=1.0)/epochs

            test_bar.set_postfix(loss=loss.item())

        save_model(model, save_dir, epoch)
    save_model(model, save_dir, epoch, last=epoch==epochs-1)

    if isinstance(report, dict):
        report["train_loss"] = running_loss
        report["val_loss"] = test_loss
        report["val_psnr"] = psnr_val
        report["val_ssim"] = ssim_val

def train_raytune(config):
    trainloader = DataLoader(config["train_dataset"], batch_size=config["batch_size"])
    testloader = DataLoader(config["test_dataset"], batch_size=config["batch_size"])
    
    model = Autoencoder(channels=config["channels"])

    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])

    report = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train(model, optimizer, config["loss_fn"], trainloader, testloader, config["epochs"], "raytune/", device=device, report=report)

    session.report(report)

def test_train():
    model = Autoencoder()
    data_dir = "/home/tkrittithee/blue-workspace/Lab6_Hyperparameter-Tuning/data"

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, file) for file in files]
    file_limit = 280
    files = files[:file_limit]
    print("File numbers: ", len(files))

    train_files, test_files = train_test_split(files, train_size=0.75, test_size=0.25)

    train_dataset = CustomImageDataset(train_files)
    test_dataset = CustomImageDataset(test_files)

    batch_size = 100
    trainloader = DataLoader(train_dataset, batch_size=batch_size)
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = dict({})
    train(model, optimizer, loss_fn, trainloader, testloader, epochs=3, save_dir="/home/tkrittithee/blue-workspace/Lab6_Hyperparameter-Tuning/model/", device=device, report=report)
    print(report)

def do_ray():
    ray.shutdown()
    ray.init(num_gpus=1,)
    # ray.init()

    data_dir = "/home/tkrittithee/blue-workspace/Lab6_Hyperparameter-Tuning/data"

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, file) for file in files]
    file_limit = 300
    files = files[:file_limit]

    train_files, test_files = train_test_split(files, train_size=0.75, test_size=0.25)

    train_dataset = CustomImageDataset(train_files)
    test_dataset = CustomImageDataset(test_files)

    config = {
        "channels": tune.grid_search([
            [32, 64, 128],
            [64, 128, 256],
            [64, 128, 256, 512],
        ]),
        "train_dataset": tune.grid_search([train_dataset]),
        "test_dataset": tune.grid_search([test_dataset]),
        "batch_size": tune.grid_search([16, 32]),
        "optimizer": tune.grid_search(["Adam", "SGD"]),
        "lr": tune.grid_search([1e-3, 8e-4, 1e-4, 1e-2]),
        "epochs": tune.grid_search([10, 50, 100]),
        "loss_fn": tune.grid_search([nn.MSELoss()]),
        # "num_samples": 10
    }
    tuner = tune.Tuner(
        tune.with_resources(
            train_raytune,
            resources={"gpu": 1, "cpu":30}
        ),
        tune_config = tune.TuneConfig(
            metric = 'val_psnr',
            mode = 'max'
        ),
        param_space = config,
    )
    result = tuner.fit()
    return result

result = do_ray()
print(result)
