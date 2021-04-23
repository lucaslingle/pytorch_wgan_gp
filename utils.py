import torch as tc
import matplotlib.pyplot as plt
import numpy as np
import uuid
import os

def compute_grad2(d_out, x_in):
    # borrowed from https://github.com/LMescheder/GAN_stability/blob/c1f64c9efeac371453065e5ce71860f4c2b97357/gan_training/train.py#L123
    batch_size = x_in.size(0)
    grad_dout = tc.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

def save_img_grid(images, grid_size, output_dir='output'):
    assert grid_size**2 == len(images)

    rows = []
    for i in range(0, grid_size):
        row = []
        for j in range(0, grid_size):
            img = images[grid_size*i+j]  # this is still in NCHW format and is in [-1, 1] range if color image.
            row.append(img)
        row = np.concatenate(row, axis=-1)
        rows.append(row)
    rows = np.concatenate(rows, axis=-2)
    rows = np.transpose(rows, [1, 2, 0]) # NHWC format

    if rows.shape[-1] == 3:
        rows = 0.5 * rows + 0.5  # convert back to [0, 1] range.

    if rows.shape[-1] == 1:
        rows = np.concatenate([rows for _ in range(3)], axis=-1)

    fn = str(uuid.uuid4()) + '.png'
    fp = os.path.join(output_dir, fn)
    plt.imsave(fname=fp, arr=rows)
    return fp