

import torch
import numpy as np
from scipy.fftpack import dctn
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm
from o2 import fc2bayer, multiresolution_dct_subband, plot_subbands


def eval_accuracy(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total = 0
    correct = 0

    progress_bar = tqdm(test_loader, desc=f"Evaluating Test Accuracy")
    model.eval()
    with torch.no_grad():
        for Ys_raw, labels in progress_bar:
            Ys_raw, labels = Ys_raw.to(device), labels.to(device)
            Ys = fc2bayer(Ys_raw.to(device))
            YmDCTs = multiresolution_dct_subband(Ys).to(device)

            # Forward pass
            outputs = model(YmDCTs)
            _, predicted = torch.max(outputs, 1)

            # loss = criterion(outputs, labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    

    acc = 100.0 * correct / total
    return acc



