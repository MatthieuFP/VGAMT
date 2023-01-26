import os
import json
import numpy as np
from torch.utils.data import DataLoader, IterableDataset, Dataset, get_worker_info
from itertools import cycle, islice


def split_dataset(params):
    if params.multimodal_model and params.mix_xp:
        train_set = [ParallelImageDataset(params, split="train"), ImageDataset(params, split="train")]
        valid_set = ParallelImageEvaluationDataset(params, split="val")
        test_set = ParallelImageEvaluationDataset(params, split="test")
    elif params.multimodal_model and not params.mix_xp:
        train_set = ParallelImageDataset(params, split="train")
        valid_set = ParallelImageEvaluationDataset(params, split="val")
        test_set = ParallelImageEvaluationDataset(params, split="test")
    elif not params.multimodal_model and params.mix_xp:
        train_set = [ParallelDataset(params, split="train"), TextDataset(params, split="train")]
        valid_set = ParallelEvaluationDataset(params, split="dev" if "multi30k" not in params.data_path else "val")
        test_set = ParallelEvaluationDataset(params, split="test")
    else:
        train_set = ParallelDataset(params, split="train")
        valid_set = ParallelEvaluationDataset(params, split="dev" if "multi30k" not in params.data_path else "val")
        test_set = ParallelEvaluationDataset(params, split="test")
    return train_set, valid_set, test_set


class TextDataset(IterableDataset):
    def __init__(self, params, split="train"):
        super(TextDataset, self).__init__()

        self.params = params
        self.split = split
        self.data_path = params.data_path if not params.mix_xp else params.data_mix_path
        self.src_text_path = os.path.join(self.data_path, f"{split}.sub.{params.src_lang}")

    def read_file(self, f_path):
        with open(f_path, "r") as f_object:
            for idx, line in enumerate(f_object):
                # Exception when reaching end of file with empty line
                if not line:
                    continue

                sample = {"text": line.strip("\n")}
                yield sample

    def get_stream(self, src_path):
        return cycle(self.read_file(src_path))

    def __iter__(self):
        worker_total_num = get_worker_info().num_workers if get_worker_info() is not None else 1
        worker_id = get_worker_info().id if get_worker_info() is not None else 0
        step = min(worker_total_num, self.params.batch_size)
        step_world_size = self.params.world_size if self.params.local_rank >= 0 else 1
        start_rank = self.params.global_rank * step if self.params.local_rank >= 0 else 0
        sample_itr = islice(self.get_stream(self.src_text_path), worker_id + start_rank, None,
                            step * step_world_size)
        return sample_itr


class ImageDataset(IterableDataset):
    def __init__(self, params, split="train"):
        super(ImageDataset, self).__init__()

        self.params = params
        self.split = split
        self.data_path = params.data_path if not params.mix_xp else params.data_mix_path
        self.src_text_path = os.path.join(self.data_path, f"{split}.sub.{params.src_lang}")
        self.image_order = open(os.path.join(self.data_path, f"{split}.sub.order")).read().split("\n")

        self.img_features_path = {}
        if len(params.features_path.split(",")) == 2:
            self.features_path = params.features_path.split(",") if not params.mix_xp else params.features_mix_path.split(",")
            self.features_path = sorted(self.features_path)
            self.img_features_path["mdetr"] = self.features_path[1]
            self.img_features_path["clip"] = self.features_path[0]
        else:
            self.img_features_path[params.features_type] = params.features_path if not params.mix_xp else params.features_mix_path

        self.features = {}
        self.load_img_features(params.features_type)

    def load_img_features(self, type="mdetr"):

        self.labels, self.boxes = None, None
        if "mdetr" in type:
            self.load_mdetr_labels()
            self.features["mdetr"] = os.path.join(self.img_features_path["mdetr"], self.split, "features")
            self.boxes = os.path.join(self.img_features_path["mdetr"], self.split, "boxes_loc")

        if "clip" in type:
            self.features["clip"] = os.path.join(self.img_features_path["clip"], self.split)

    def load_mdetr_labels(self):
        labels_path = os.path.join(self.img_features_path["mdetr"], self.split, "labels", "labels.json")
        with open(labels_path, "r") as fj:
            self.labels = json.load(fj)
        self.labels = {fname.replace(".txt", ""): v for fname, v in self.labels.items()}

    def read_file(self, f_path):
        with open(f_path, "r") as f_object:
            for idx, line in enumerate(f_object):
                # Exception when reaching end of file with empty line
                if not line:
                    continue

                im_name = self.image_order[idx].split("/")[-1]
                img_sample = {"clip_features": None,
                              "mdetr_features": None,
                              "boxes_loc": None,
                              "labels": None}

                if "mdetr" in self.params.features_type:
                    features = np.load(os.path.join(self.features["mdetr"], im_name + ".npy"))
                    boxes = np.load(os.path.join(self.boxes, im_name + ".npy"))
                    label = self.labels[im_name.split(".")[0]]
                    img_sample["mdetr_features"] = features
                    img_sample["boxes_loc"] = boxes
                    img_sample["labels"] = label

                if "clip" in self.params.features_type:
                    features = np.load(os.path.join(self.features["clip"], im_name + ".npy"))
                    img_sample["clip_features"] = features

                sample = {"text": line.strip("\n")}
                sample.update(img_sample)

                yield sample

    def get_stream(self, src_path):
        return cycle(self.read_file(src_path))

    def __iter__(self):
        worker_total_num = get_worker_info().num_workers if get_worker_info() is not None else 1
        worker_id = get_worker_info().id if get_worker_info() is not None else 0
        step = min(worker_total_num, self.params.batch_size)
        step_world_size = self.params.world_size if self.params.local_rank >= 0 else 1
        start_rank = self.params.global_rank * step if self.params.local_rank >= 0 else 0
        sample_itr = islice(self.get_stream(self.src_text_path), worker_id + start_rank, None,
                            step * step_world_size)
        return sample_itr


class ParallelImageDataset(IterableDataset):
    def __init__(self, params, split="train"):
        super(ParallelImageDataset, self).__init__()

        self.params = params
        self.src_text_path = os.path.join(params.data_path, f"{split}.{params.src_lang}")
        self.tgt_text_path = os.path.join(params.data_path, f"{split}.{params.tgt_lang}")
        self.image_order = open(os.path.join(params.data_path, f"{split}.order")).read().split("\n")

        self.img_features_path = {}
        if len(params.features_path.split(",")) == 2:
            self.features_path = params.features_path.split(",")
            self.features_path = sorted(self.features_path)
            self.img_features_path["mdetr"] = self.features_path[1]
            self.img_features_path["clip"] = self.features_path[0]
        else:
            self.img_features_path[params.features_type] = params.features_path

        self.features = {}
        self.load_img_features(params.features_type)

    def load_img_features(self, type="mdetr"):

        self.labels, self.boxes = None, None
        if "mdetr" in type:
            self.load_mdetr_labels()
            self.features["mdetr"] = os.path.join(self.img_features_path["mdetr"], "features")
            self.boxes = os.path.join(self.img_features_path["mdetr"], "boxes_loc")

        if "clip" in type:
            self.features["clip"] = self.img_features_path["clip"]

    def load_mdetr_labels(self):
        labels_path = os.path.join(self.img_features_path["mdetr"], "labels")
        self.labels = {fname.replace(".txt", ""): open(os.path.join(labels_path, fname), "r").read().strip("\n")
                       for fname in os.listdir(labels_path) if fname.endswith(".txt")}

    def read_file(self, f_path1, f_path2):
        with open(f_path1, "r") as f1_object, open(f_path2, "r") as f2_object:
            for idx, (line1, line2) in enumerate(zip(f1_object, f2_object)):
                # Exception when reaching end of file with empty line
                if not line1:
                    continue

                im_name = self.image_order[idx]
                img_sample = {"clip_features": None,
                              "mdetr_features": None,
                              "boxes_loc": None,
                              "labels": None}

                if "mdetr" in self.params.features_type:
                    features = np.load(os.path.join(self.features["mdetr"], im_name + ".npy"))
                    boxes = np.load(os.path.join(self.boxes, im_name + ".npy"))
                    label = self.labels[im_name.split(".")[0]]
                    img_sample["mdetr_features"] = features
                    img_sample["boxes_loc"] = boxes
                    img_sample["labels"] = label
                    
                if "clip" in self.params.features_type:
                    features = np.load(os.path.join(self.features["clip"], im_name + ".npy"))
                    img_sample["clip_features"] = features

                sample = {self.params.src_lang: line1.strip("\n"), self.params.tgt_lang: line2.strip("\n")}
                sample.update(img_sample)

                yield sample

    def get_stream(self, src_path, tgt_path):
        return cycle(self.read_file(src_path, tgt_path))

    def __iter__(self):
        worker_total_num = get_worker_info().num_workers if get_worker_info() is not None else 1
        worker_id = get_worker_info().id if get_worker_info() is not None else 0
        step = min(worker_total_num, self.params.batch_size)
        step_world_size = self.params.world_size if self.params.local_rank >= 0 else 1
        start_rank = self.params.global_rank * step if self.params.local_rank >= 0 else 0
        sample_itr = islice(self.get_stream(self.src_text_path, self.tgt_text_path), worker_id + start_rank, None, step * step_world_size)
        return sample_itr


class ParallelDataset(IterableDataset):
    def __init__(self, params, split="train"):
        super(ParallelDataset, self).__init__()

        self.params = params

        if "multi30k" not in params.data_path:
            self.data_path = os.path.join(params.data_path, f"{split}.{params.src_lang}{params.tgt_lang}")
        else:
            self.data_path = params.data_path
            self.src_text_path = os.path.join(params.data_path, f"{split}.{params.src_lang}")
            self.tgt_text_path = os.path.join(params.data_path, f"{split}.{params.tgt_lang}")

    def read_file(self, f_path):
        with open(f_path, "r") as f_object:
            for line in f_object:
                # Exception when reaching end of file with empty line
                if not line:
                    continue
                try:
                    sample = [line.strip("\n").split("\t")]
                except:
                    continue
                yield from sample

    def read_files(self, src_path, tgt_path):
        with open(src_path, "r") as src_object, open(tgt_path, "r") as tgt_object:
            for idx, (src_inp, tgt_inp) in enumerate(zip(src_object, tgt_object)):
                # Exception when reaching end of file with empty line
                if not src_inp:
                    continue

                sample = [(src_inp.strip("\n"), tgt_inp.strip("\n"))]
                yield from sample

    def get_stream(self):
        if "multi30k" not in self.data_path:
            return cycle(self.read_file(self.data_path))
        else:
            return cycle(self.read_files(self.src_text_path, self.tgt_text_path))

    def __iter__(self):
        worker_total_num = get_worker_info().num_workers if get_worker_info() is not None else 1
        worker_id = get_worker_info().id if get_worker_info() is not None else 0
        step = min(worker_total_num, self.params.batch_size)
        step_world_size = self.params.world_size if self.params.local_rank >= 0 else 1
        start_rank = self.params.global_rank * step if self.params.local_rank >= 0 else 0
        sample_itr = islice(self.get_stream(), worker_id + start_rank, None, step * step_world_size)
        return sample_itr


class ParallelEvaluationDataset(Dataset):
    def __init__(self, params, split="dev"):
        super(ParallelEvaluationDataset, self).__init__()

        self.params = params
        self.is_m30k = "multi30k" in params.data_path
        if not self.is_m30k:
            self.data_path = os.path.join(params.data_path, f"{split}.{params.src_lang}{params.tgt_lang}")
            self.read_file()
        else:
            if split == "test" and self.params.test_data_set:
                split = self.params.test_data_set
            self.src_text_data = open(os.path.join(params.data_path, f"{split}.{params.src_lang}"), "r").read().split(
                "\n")
            self.tgt_text_data = open(os.path.join(params.data_path, f"{split}.{params.tgt_lang}"), "r").read().split(
                "\n")

            if not self.src_text_data[-1]:
                self.src_text_data = self.src_text_data[:-1]
            if not self.tgt_text_data[-1]:
                self.tgt_text_data = self.tgt_text_data[:-1]

    def read_file(self):
        with open(self.data_path, "r") as f:
            self.data = f.read().split("\n")[:-1]

    def __len__(self):
        return len(self.data) if not self.is_m30k else len(self.src_text_data)

    def __getitem__(self, item):

        if not self.is_m30k:
            src_item, tgt_item = self.data[item].strip("\n").split("\t")
        else:
            src_item, tgt_item = self.src_text_data[item], self.tgt_text_data[item]
            
        return {self.params.src_lang: src_item, self.params.tgt_lang: tgt_item}


class ParallelImageEvaluationDataset(Dataset):
    def __init__(self, params, split="dev"):
        super(ParallelImageEvaluationDataset, self).__init__()

        self.params = params
        if split == "test" and self.params.test_data_set:
            split = self.params.test_data_set
            
        self.src_text_data = open(os.path.join(params.data_path, f"{split}.{params.src_lang}"), "r").read().split("\n")
        self.tgt_text_data = open(os.path.join(params.data_path, f"{split}.{params.tgt_lang}"), "r").read().split("\n")
        self.image_order = open(os.path.join(params.data_path, f"{split}.order"), "r").read().split("\n")
        self.rm_empty_last_lines()

        self.img_features_path = {}
        if len(params.features_path.split(",")) == 2:
            self.features_path = params.features_path.split(",")
            self.features_path = sorted(self.features_path)
            self.img_features_path["mdetr"] = self.features_path[1]
            self.img_features_path["clip"] = self.features_path[0]
        else:
            self.img_features_path[params.features_type] = params.features_path

        self.features = {}
        self.load_img_features(params.features_type)

    def load_img_features(self, type="mdetr"):

        self.labels, self.boxes = None, None
        if "mdetr" in type:
            self.load_mdetr_labels()
            self.features["mdetr"] = os.path.join(self.img_features_path["mdetr"], "features")
            self.boxes = os.path.join(self.img_features_path["mdetr"], "boxes_loc")

        if "clip" in type:
            self.features["clip"] = self.img_features_path["clip"]

    def load_mdetr_labels(self):
        labels_path = os.path.join(self.img_features_path["mdetr"], "labels")
        self.labels = {fname.replace(".txt", ""): open(os.path.join(labels_path, fname), "r").read().strip("\n")
                       for fname in os.listdir(labels_path) if fname.endswith(".txt")}

    def __len__(self):
        return len(self.src_text_data)

    def __getitem__(self, item):

        src_text, tgt_text = self.src_text_data[item], self.tgt_text_data[item]
        im_name = self.image_order[item]
        img_sample = {"clip_features": None,
                      "mdetr_features": None,
                      "boxes_loc": None,
                      "labels": None}

        if "mdetr" in self.params.features_type:
            features = np.load(os.path.join(self.features["mdetr"], im_name + ".npy"))
            boxes = np.load(os.path.join(self.boxes, im_name + ".npy"))
            label = self.labels[im_name.split(".")[0]]
            img_sample["mdetr_features"] = features
            img_sample["boxes_loc"] = boxes
            img_sample["labels"] = label

        if "clip" in self.params.features_type:
            features = np.load(os.path.join(self.features["clip"], im_name + ".npy"))
            img_sample["clip_features"] = features

        sample = {self.params.src_lang: src_text.strip("\n"), self.params.tgt_lang: tgt_text.strip("\n")}
        sample.update(img_sample)

        return sample

    def rm_empty_last_lines(self):
        if not self.src_text_data[-1]:
            self.src_text_data = self.src_text_data[:-1]
        if not self.tgt_text_data[-1]:
            self.tgt_text_data = self.tgt_text_data[:-1]
        if not self.image_order[-1]:
            self.image_order = self.image_order[:-1]
        assert len(self.src_text_data) == len(self.tgt_text_data) == len(self.image_order)


class ImageEvaluationDataset(Dataset):
    def __init__(self, params, split="dev"):
        super(ImageEvaluationDataset, self).__init__()

        self.params = params
        self.split = split

        self.src_text_data = open(os.path.join(params.data_path, f"{split}.sub.{params.src_lang}"), "r").read().split("\n")
        self.image_order = open(os.path.join(params.data_path, f"{split}.sub.order"), "r").read().split("\n")
        self.rm_empty_last_lines()

        self.img_features_path = {}
        if len(params.features_path.split(",")) == 2:
            self.features_path = params.features_path.split(",")
            self.features_path = sorted(self.features_path)
            self.img_features_path["mdetr"] = self.features_path[1]
            self.img_features_path["clip"] = self.features_path[0]
        else:
            self.img_features_path[params.features_type] = params.features_path

        self.features = {}
        self.load_img_features(params.features_type)

    def load_img_features(self, type="mdetr"):

        self.labels, self.boxes = None, None
        if "mdetr" in type:
            self.load_mdetr_labels()
            self.features["mdetr"] = os.path.join(self.img_features_path["mdetr"], self.split, "features")
            self.boxes = os.path.join(self.img_features_path["mdetr"], self.split, "boxes_loc")

        if "clip" in type:
            self.features["clip"] = os.path.join(self.img_features_path["clip"], self.split)

    def load_mdetr_labels(self):
        labels_path = os.path.join(self.img_features_path["mdetr"], self.split, "labels")
        self.labels = {fname.replace(".txt", ""): open(os.path.join(labels_path, fname), "r").read().strip("\n")
                       for fname in os.listdir(labels_path) if fname.endswith(".txt")}

    def __len__(self):
        return len(self.src_text_data)

    def __getitem__(self, item):

        src_text = self.src_text_data[item]
        im_name = self.image_order[item].split("/")[-1]
        img_sample = {"clip_features": None,
                      "mdetr_features": None,
                      "boxes_loc": None,
                      "labels": None}

        if "mdetr" in self.params.features_type:
            features = np.load(os.path.join(self.features["mdetr"], im_name + ".npy"))
            boxes = np.load(os.path.join(self.boxes, im_name + ".npy"))
            label = self.labels[im_name.split(".")[0]]
            img_sample["mdetr_features"] = features
            img_sample["boxes_loc"] = boxes
            img_sample["labels"] = label

        if "clip" in self.params.features_type:
            features = np.load(os.path.join(self.features["clip"], im_name + ".npy"))
            img_sample["clip_features"] = features

        sample = {"text": src_text.strip("\n")}
        sample.update(img_sample)

        return sample

    def rm_empty_last_lines(self):
        if not self.src_text_data[-1]:
            self.src_text_data = self.src_text_data[:-1]
        if not self.image_order[-1]:
            self.image_order = self.image_order[:-1]

        assert len(self.src_text_data) == len(self.image_order)


class TextEvaluationDataset(Dataset):
    def __init__(self, params, split="dev"):
        super(TextEvaluationDataset, self).__init__()

        self.params = params
        self.split = split

        self.src_text_data = open(os.path.join(params.data_path, f"{split}.sub.{params.src_lang}"), "r").read().split("\n")
        self.rm_empty_last_lines()

    def __len__(self):
        return len(self.src_text_data)

    def __getitem__(self, item):
        src_text = self.src_text_data[item]
        sample = {"text": src_text.strip("\n")}
        return sample

    def rm_empty_last_lines(self):
        if not self.src_text_data[-1]:
            self.src_text_data = self.src_text_data[:-1]


class CommuteEvaluationDataset(Dataset):
    def __init__(self, params):
        super(CommuteEvaluationDataset, self).__init__()

        self.params = params

        self.src_text_data = open(os.path.join(params.data_path, f"src.{params.src_lang}"), "r").read().split("\n")
        self.tgt_correct_text_data = open(os.path.join(params.data_path, f"correct.{params.tgt_lang}"),
                                          "r").read().split("\n")
        self.tgt_incorrect_text_data = open(os.path.join(params.data_path, f"incorrect.{params.tgt_lang}"),
                                          "r").read().split("\n")
        self.image_order = open(os.path.join(params.data_path, f"img.order"), "r").read().split("\n")
        self.rm_empty_last_lines()

        self.img_features_path = {}
        if len(params.features_path.split(",")) == 2:
            self.features_path = params.features_path.split(",")
            self.features_path = sorted(self.features_path)
            self.img_features_path["mdetr"] = self.features_path[1]
            self.img_features_path["clip"] = self.features_path[0]
        else:
            self.img_features_path[params.features_type] = params.features_path

        self.features = {}
        self.load_img_features(params.features_type)

    def load_img_features(self, type="mdetr"):

        self.labels, self.boxes = None, None
        if "mdetr" in type:
            self.load_mdetr_labels()
            self.features["mdetr"] = os.path.join(self.img_features_path["mdetr"], "features")
            self.boxes = os.path.join(self.img_features_path["mdetr"], "boxes_loc")

        if "clip" in type:
            self.features["clip"] = self.img_features_path["clip"]

    def load_mdetr_labels(self):
        labels_path = os.path.join(self.img_features_path["mdetr"], "labels")
        self.labels = {fname.replace(".txt", ""): open(os.path.join(labels_path, fname), "r").read().strip("\n")
                       for fname in os.listdir(labels_path) if fname.endswith(".txt")}

    def __len__(self):
        return len(self.src_text_data)

    def __getitem__(self, item):

        src_text, correct_tgt_text, incorrect_tgt_text = self.src_text_data[item], self.tgt_correct_text_data[item], \
                                                         self.tgt_incorrect_text_data[item]
        im_name = self.image_order[item]
        img_sample = {"clip_features": None,
                      "mdetr_features": None,
                      "boxes_loc": None,
                      "labels": None}

        if "mdetr" in self.params.features_type:
            features = np.load(os.path.join(self.features["mdetr"], im_name + ".npy"))
            boxes = np.load(os.path.join(self.boxes, im_name + ".npy"))
            label = self.labels[im_name.split(".")[0]]
            img_sample["mdetr_features"] = features
            img_sample["boxes_loc"] = boxes
            img_sample["labels"] = label

        if "clip" in self.params.features_type:
            features = np.load(os.path.join(self.features["clip"], im_name + ".npy"))
            img_sample["clip_features"] = features

        if not self.params.commute_generation:
            sample = {self.params.src_lang: src_text.strip("\n"),
                      f"{self.params.tgt_lang}_correct": correct_tgt_text.strip("\n"),
                      f"{self.params.tgt_lang}_incorrect": incorrect_tgt_text.strip("\n")}
        else:
            sample = {self.params.src_lang: src_text.strip("\n"),
                      f"{self.params.tgt_lang}_correct": correct_tgt_text.strip("\n")}

        sample.update(img_sample)

        return sample

    def rm_empty_last_lines(self):
        if not self.src_text_data[-1]:
            self.src_text_data = self.src_text_data[:-1]
        if not self.tgt_correct_text_data[-1]:
            self.tgt_correct_text_data = self.tgt_correct_text_data[:-1]
        if not self.tgt_incorrect_text_data[-1]:
            self.tgt_incorrect_text_data = self.tgt_incorrect_text_data[:-1]
        if not self.image_order[-1]:
            self.image_order = self.image_order[:-1]
        assert len(self.src_text_data) == len(self.tgt_correct_text_data) \
               == len(self.tgt_incorrect_text_data) == len(self.image_order)


class CommuteEvaluationTextOnlyDataset(Dataset):
    def __init__(self, params):
        super(CommuteEvaluationTextOnlyDataset, self).__init__()

        self.params = params

        self.src_text_data = open(os.path.join(params.data_path, f"src.{params.src_lang}"), "r").read().split("\n")
        self.tgt_correct_text_data = open(os.path.join(params.data_path, f"correct.{params.tgt_lang}"),
                                          "r").read().split("\n")
        self.tgt_incorrect_text_data = open(os.path.join(params.data_path, f"incorrect.{params.tgt_lang}"),
                                          "r").read().split("\n")
        self.rm_empty_last_lines()

    def __len__(self):
        return len(self.src_text_data)

    def __getitem__(self, item):

        src_text, correct_tgt_text, incorrect_tgt_text = self.src_text_data[item], self.tgt_correct_text_data[item], \
                                                         self.tgt_incorrect_text_data[item]

        sample = {self.params.src_lang: src_text.strip("\n"),
                  f"{self.params.tgt_lang}_correct": correct_tgt_text.strip("\n")}

        return sample

    def rm_empty_last_lines(self):
        if not self.src_text_data[-1]:
            self.src_text_data = self.src_text_data[:-1]
        if not self.tgt_correct_text_data[-1]:
            self.tgt_correct_text_data = self.tgt_correct_text_data[:-1]
        if not self.tgt_incorrect_text_data[-1]:
            self.tgt_incorrect_text_data = self.tgt_incorrect_text_data[:-1]
        assert len(self.src_text_data) == len(self.tgt_correct_text_data) \
               == len(self.tgt_incorrect_text_data)
