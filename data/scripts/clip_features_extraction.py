# -*- coding: utf-8 -*-

"""
Created on 24/01/2022

@author: matthieufuteral-peter
"""

import os
import io
from io import BytesIO
import tarfile
from logger import logger
import numpy as np
import clip
from tqdm import tqdm
import argparse
import pickle
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from PIL import Image
from typing import Iterable, Dict, Callable
from multiprocessing import Pool
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info


class ImageDataset(Dataset):
    def __init__(self, params, data_path, img_index):
        super(ImageDataset, self).__init__()

        self.params = params
        self.dataset_root = os.path.basename(data_path)
        self.data_path = os.path.dirname(data_path)
        if "commute" in data_path:
            self.data_path = data_path
        self.img_index = img_index
        if not self.img_index[-1]:
            self.img_index = self.img_index[:-1]

    def __len__(self):
        return len(self.img_index)

    def __getitem__(self, item):
        im_name = self.img_index[item]
        if im_name.startswith("train") or "commute" in self.data_path:
            path_image = os.path.join(self.data_path, im_name)
        else:
            path_image = os.path.join(self.data_path, self.dataset_root, im_name)

        path_dump = os.path.join(self.params.dump_root, im_name + ".npy")
        try:
            image = Image.open(path_image)
            img = preprocess(image)
        except:
            img = torch.zeros((3, 224, 224))
            fail_name = os.path.basename(im_name)
            path_dump = os.path.join(args.dump_root, "fail/" + fail_name + ".npy")
        return {"image": img, "dump_path": path_dump}


def fn_pickle(feat, fname):
    with open(fname, 'wb') as f:
        pickle.dump(feat, f, protocol=4, fix_imports=False)


def fn_numpy(feats, fnames):
    for idx, fname in enumerate(fnames):
        np.save(fname, feats[idx])


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model.visual
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
        return self._features


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-i", "--img_root", type=str, default="./multi30k/images")
    parser.add_argument("-d", "--dump_root", type=str, default="./multi30k/clip_features")
    parser.add_argument('-l', '--list-of-images', type=str, default="./multi30k/images/index.txt")
    parser.add_argument('-t', '--tarball', action='store_true', help='img root is a tarball')
    parser.add_argument('--num_list', type=str, default='')
    parser.add_argument('-p', '--parallel', action='store_true', help='Parallel dumper process for output files.')
    parser.add_argument('--cls', action="store_true", help="Use the cls token")
    args = parser.parse_args()

    if args.parallel:
        pool = Pool(processes=4)
    else:
        pool = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess = clip.load('ViT-B/32', device, download_root=os.path.join(os.environ.get("STORE"), "clip"))
    if not args.cls:
        feature_extractor = FeatureExtractor(model, layers=["transformer"])

    with open(args.list_of_images, "r") as f:
        img_list = f.read().split("\n")

    if not img_list[-1]:
        img_list = img_list[:-1]

    if args.num_list:
        chunk = len(img_list) // 100
        img_list = img_list[int(args.num_list) * chunk: (int(args.num_list) + 1) * chunk]

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


