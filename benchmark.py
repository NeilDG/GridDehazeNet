from model import GridDehazeNet
import glob
import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torchvision.utils as utils
import argparse
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (indoor or outdoor?)', default='indoor', type=str)
args = parser.parse_args()

network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category

def normalize_to_matplotimg(img_tensor, batch_idx, std, mean):
    img = img_tensor[batch_idx, :, :, :].numpy()
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0)  # for properly displaying image in matplotlib

    img = ((img * std) + mean)  # normalize back to 0-1 range

    img = cv2.convertScaleAbs(img, alpha=(255.0))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def benchmark_reside():
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform_op = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize((512, 512)),
                                       transforms.CenterCrop((512, 512)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    HAZY_PATH = "E:/Hazy Dataset Benchmark/RESIDE-Unannotated/"
    SAVE_PATH = "results_reside/"

    hazy_list = glob.glob(HAZY_PATH + "*.jpg")
    for i, (hazy_path) in enumerate(hazy_list):
        with torch.no_grad():
            img_name = hazy_path.split("\\")[1]
            hazy_img = cv2.imread(hazy_path)
            hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)

            hazy_tensor = torch.unsqueeze(transform_op(hazy_img), 0).to(device)

            net = GridDehazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer,
                                growth_rate=growth_rate)

            # --- Multi-GPU --- #
            net = net.to(device)
            net = nn.DataParallel(net, device_ids=device_ids)

            # --- Load the network weight --- #
            net.load_state_dict(torch.load('indoor_haze_best_3_6'))
            net.eval()

            result_tensor = net(hazy_tensor).cpu()
            utils.save_image(result_tensor, SAVE_PATH + img_name)

def benchmark():
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform_op = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize((512,512)),
                                       transforms.CenterCrop((512,512)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    HAZY_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/hazy/"
    #HAZY_PATH = "data/test/SOTS/indoor/hazy/"
    GT_PATH = "E:/Hazy Dataset Benchmark/O-HAZE/GT/"
    SAVE_PATH = "results/"
    BENCHMARK_PATH = "results/metrics.txt"

    hazy_list = glob.glob(HAZY_PATH + "*.jpg")
    gt_list = glob.glob(GT_PATH + "*.jpg")
    average_SSIM = 0.0
    for i, (hazy_path, gt_path) in enumerate(zip(hazy_list, gt_list)):
        with torch.no_grad():
            img_name = hazy_path.split("\\")[1]
            hazy_img = cv2.imread(hazy_path)
            hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
            clear_img = cv2.imread(gt_path)
            clear_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
            clear_img = cv2.resize(clear_img, (512,512), interpolation=cv2.INTER_CUBIC)

            hazy_tensor = torch.unsqueeze(transform_op(hazy_img), 0).to(device)

            net = GridDehazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer,
                                growth_rate=growth_rate)

            # --- Multi-GPU --- #
            net = net.to(device)
            net = nn.DataParallel(net, device_ids=device_ids)

            # --- Load the network weight --- #
            net.load_state_dict(torch.load('indoor_haze_best_3_6'))
            net.eval()

            result_tensor = net(hazy_tensor).cpu()
            utils.save_image(result_tensor, SAVE_PATH + img_name)

            # measure SSIM
            result_img = cv2.imread(SAVE_PATH + img_name)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            SSIM = np.round(compare_ssim(result_img, clear_img, multichannel=True), 4)
            print("SSIM of " + hazy_path + " : ", SSIM)
            average_SSIM += SSIM

    average_SSIM = np.round(average_SSIM / len(hazy_list) * 1.0, 4)
    print("Average SSIM: ", average_SSIM)

def main():
    #benchmark()
    benchmark_reside()

if __name__=="__main__":
    main()