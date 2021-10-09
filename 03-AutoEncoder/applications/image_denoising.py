import argparse
import torch
import sys

parser = argparse.ArgumentParser("Denoising Images using AE")
parser.add_argument("--model_path", default="/checkpoint/ae.pth", type=str, help="Path to model's weight")
parser.add_argument( 
    "--main_path",
    default="/home/lukedinh/Desktop/Unsupervised-Learning/03-AutoEncoder",
    type=str,
    help="Path to your folder"
)
opt = parser.parse_args()
main_path = opt.main_path
model_path = opt.model_path
sys.path.append(main_path)

# Load model
from model.ae import AE
model = AE(in_dims=784, encod_dims=64)
model.load_state_dict(torch.load(main_path + model_path, map_location="cpu")["state_dict"])
model.eval()