import os, shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import copy

def generate_img_batch(syn_batch, ref_batch, real_batch, png_path, ref_preds=None, real_preds=None):
    '''
    Generates matplotlib plots of 3 syn, ref, and real examples from the step.
    '''
    # # syn_batch_type: Tensor, ref_batch_type: Tensor, real_batch_type: Tensor
    #syn_batch = syn_batch[:5]
    #ref_batch = ref_batch[:5]
    #real_batch = real_batch[:5]

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(1, 3, 1)
    for i in range(5):
        ax.plot(syn_batch[i][0].numpy())
    ax.set_ylim([0, 1])
    ax.set_title("Simulated Signals")
    ax = fig.add_subplot(1, 3, 2)
    for i in range(5):
        if ref_preds is not None:
            ref_probs = torch.sigmoid(ref_preds[i]).item()
            label = str([1 - ref_probs, ref_probs])
        else:
            label=None
        ax.plot(ref_batch[i][0].numpy(), label=label)
    ax.set_ylim([0, 1])
    ax.set_title("Refined Signals")
    plt.legend()
    ax = fig.add_subplot(1, 3, 3)
    for i in range(5):
        if real_preds is not None:
            real_probs = torch.sigmoid(real_preds[i]).item()
            label = str([1 - real_probs, real_probs])
        else:
            label=None
        ax.plot(real_batch[i][0].numpy(), label=label)
    plt.legend()
    ax.set_ylim([0, 1])
    ax.set_title("Random Real Signals")
    plt.tight_layout()

    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    img = Image.fromarray(img)

    img.save(png_path, 'png')

    plt.close()