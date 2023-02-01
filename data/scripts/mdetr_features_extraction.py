# -*- coding: utf-8 -*-

"""
Created on 21/02/2022

@author: matthieufuteral-peter
"""
import json
import os
import pickle
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
from imgcat import imgcat

torch.set_grad_enabled(False)


def fn_pickle(out, fname):
    with open(fname, 'wb') as f:
        pickle.dump(out, f, protocol=4, fix_imports=False)


def save_all(out, save_name, save_name_txt):
    np.save(os.path.join(save_path_boxes, save_name), out["boxes_loc"])
    np.save(os.path.join(save_path_features, save_name), out["features"])
    np.save(os.path.join(save_path_last_hidden_states, save_name), out["last_hidden_states"])


def print_image(path):
    return imgcat(Image.open(path))


transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img_root", type=str, default="./conceptual_captions/images/train")
    parser.add_argument("-d", "--dump_root", type=str, default="./conceptual_captions/features/mdetr_features/train")
    parser.add_argument('-l', '--list-of-images', type=str, default="./conceptual_captions/train.order")
    parser.add_argument('--text', type=str, default="./conceptual_captions/train.en")
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-p', '--parallel', action='store_true', help='Parallel dumper process for output files.')
    args = parser.parse_args()

    if args.parallel:
        pool = Pool(processes=4)
    else:
        pool = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True,
                                          return_postprocessor=True)

    model.to(device)
    model.eval()

    with open(args.list_of_images, "r") as f:
        img_list = f.read().strip("\n").split("\n")

    with open(args.text, "r") as f:
        txt_list = f.read().strip("\n").split("\n")

    txt_list = [txt.lower() for txt in txt_list]

    assert len(img_list) == len(txt_list)
    no_bb_detected = 0

    save_path_boxes = os.path.join(args.dump_root, "boxes_loc")
    save_path_features = os.path.join(args.dump_root, "features")
    save_path_labels = os.path.join(args.dump_root, "labels")
    save_path_last_hidden_states = os.path.join(args.dump_root, "last_hidden_states")

    os.makedirs(save_path_boxes, exist_ok=True)
    os.makedirs(save_path_features, exist_ok=True)
    os.makedirs(save_path_labels, exist_ok=True)
    os.makedirs(save_path_last_hidden_states, exist_ok=True)

    labels = {}
    for idx, (im_name, caption) in tqdm(enumerate(zip(img_list, txt_list), 1)):
        path_img = os.path.join(args.img_root, im_name)
        save_name = im_name + ".npy" if "conceptual_captions" not in args.dump_root else im_name.split("/")[-1] + ".npy"
        save_name_txt = im_name.split(".")[0] + ".txt" if "conceptual_captions" not in args.dump_root else im_name.split(".")[0].split("/")[-1] + ".txt"

        try:
            im = Image.open(path_img)
        except:
            no_bb_detected += 1
            continue

        try:
            img = transform(im).unsqueeze(0).cuda()
        except:
            print(f"WARNING ! {im_name} is a gray color image")
            img = torch.zeros((1, 3, 800, 800)).cuda()

        with torch.no_grad():
            # propagate through the model
            memory_cache = model(img, [caption], encode_and_save=True)
            outputs = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

        # keep only predictions with 0.7+ confidence
        probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
        keep = (probas > args.threshold).cpu()
        scores = probas[keep].cpu().numpy()

        if not keep.sum().item():
            no_bb_detected += 1
            print(f"{caption} \t - \t {im_name}")

        proj_queries = outputs['proj_queries'].cpu()[0, keep].numpy()  # Features to save
        # convert boxes from [0; 1] to image scales
        bboxes = outputs['pred_boxes'].cpu()[0, keep].numpy()  # No rescaling
        last_hidden_states = outputs["last_hidden_state"].cpu()[0, keep].numpy()

        # Extract the text spans predicted by each box
        positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()

        # Check one text span for each box - otherwise empty text
        positions = [tok[0] for tok in positive_tokens]
        empty_text_pos = [i for i in range(bboxes.shape[0]) if i not in positions]
        for pos in empty_text_pos:
            positive_tokens.append([pos, 255])
        predicted_spans = {}
        for tok in positive_tokens:
            item, pos = tok
            if item not in predicted_spans.keys():
                predicted_spans[item] = []
            sent_tokenized = memory_cache["tokenized"]['input_ids'][0].cpu()
            if pos < 255:
                predicted_spans[item].append(sent_tokenized[pos])

        labels = [model.transformer.tokenizer.decode(predicted_spans[k]) for k in sorted(list(predicted_spans.keys()))]
        labels = [lab.strip() for lab in labels]
        idx = [1 if lab else 0 for lab in labels]
        labels = [lab for i, lab in zip(idx, labels) if i]

        _idx = torch.where(torch.LongTensor(idx) == 1)

        out = {
                "features": proj_queries[_idx, :] if proj_queries.shape[0] == 1 and not _idx[0].size(0) == 0 or _idx[0].size(0) == 1 else proj_queries[_idx],
                "boxes_loc": bboxes[_idx, :] if bboxes.shape[0] == 1 and not _idx[0].size(0) == 0 or _idx[0].size(0) == 1 else bboxes[_idx],
                "scores": scores[_idx][None] if scores.shape == (1,) and not _idx[0].size(0) == 0 or _idx[0].size(0) == 1 else scores[_idx],
                "last_hidden_states": last_hidden_states[_idx, :] if last_hidden_states.shape[0] == 1 and not _idx[0].size(0) == 0 or _idx[0].size(0) == 1 else proj_queries[_idx]
            }

        labels[save_name_txt.split(".")[0]] = "\n".join(labels).strip("\n")

        if args.parallel:
            pool.apply_async(save_all, (out, save_name, save_name_txt))
        else:
            save_all(out, save_name, save_name_txt)

    with open(os.path.join(save_path_labels, "labels.json"), "w") as fj:
        json.dump(labels, fj)

    print(f"No detected bb in {no_bb_detected}/{len(img_list)}")
