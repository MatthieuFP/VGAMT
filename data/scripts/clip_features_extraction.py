# -*- coding: utf-8 -*-

"""
Created on 24/01/2022

@author: matthieufuteral-peter
"""

import os
import numpy as np
import clip
from tqdm import tqdm
import argparse
import pickle
import torch
from torch.cuda.amp import autocast
from PIL import Image
from multiprocessing import Pool
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info


class ImageDataset(Dataset):
    def __init__(self, params, data_path, img_index):
        super(ImageDataset, self).__init__()

        self.params = params
        self.data_path = data_path
        self.img_index = img_index

    def __len__(self):
        return len(self.img_index)

    def __getitem__(self, item):

        im_name = self.img_index[item]
        path_image = os.path.join(self.data_path, im_name)
        path_dump = os.path.join(self.params.dump_root, im_name + ".npy")

        try:
            image = Image.open(path_image)
            img = preprocess(image)
        except:
            img = torch.zeros((3, 224, 224))
            fail_name = os.path.basename(im_name)
            path_dump = os.path.join(args.dump_root, "fail/" + fail_name + ".npy")
        return {"image": img, "dump_path": path_dump}


def fn_numpy(feats, fnames):
    for idx, fname in enumerate(fnames):
        np.save(fname, feats[idx])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-i", "--img_root", type=str, default="./multi30k/images",
                        help="Path where images are stored")
    parser.add_argument("-d", "--dump_root", type=str, default="./multi30k/clip_features",
                        help="Path where to save CLIP features")
    parser.add_argument('-l', '--list-of-images', type=str, default="./multi30k/train.order",
                        help="list of image names")
    parser.add_argument('-p', '--parallel', action='store_true',
                        help='Parallel dumper process for output files.')
    args = parser.parse_args()

    if args.parallel:
        pool = Pool(processes=4)
    else:
        pool = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load('ViT-B/32', device)

    with open(args.list_of_images, "r") as f:
        img_list = f.read().strip("\n").split("\n")

    if not os.path.isdir(args.dump_root):
        os.mkdir(args.dump_root)

    data = ImageDataset(args, args.img_root, img_list)
    dataloader = DataLoader(data, batch_size=args.batch_size, num_workers=10, shuffle=False)

    for batch in tqdm(dataloader):
        imgs = batch["image"].to(device)
        dump_path = batch["dump_path"]

        with torch.no_grad():
            with autocast():
                out = model.encode_image(imgs).cpu().squeeze(1).detach().numpy()

        if args.parallel:
            pool.apply_async(fn_numpy, (out, dump_path))
        else:
            fn_numpy(out, dump_path)


