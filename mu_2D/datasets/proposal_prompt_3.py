import os
import pickle
from collections import OrderedDict
import pandas as pd

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class Proposal_prompt_3(DatasetBase):
    dataset_name = "proposal_prompt_3"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, 'proposal_prompt_imgs')
        # self.image_dir = os.path.join(self.dataset_dir, "images")

        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")

        # self.preprocessed = os.path.join(self.dataset_dir.replace('group', 'sheng'), "preprocessed.pkl")
        # self.split_fewshot_dir = os.path.join(self.dataset_dir.replace('group', 'sheng'), "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                val = preprocessed["val"]
                test = preprocessed["test"]
        else:
            # text_file = os.path.join(self.dataset_dir, "classnames.txt")

            # HACK: hack for trevor's group machine dir's
            text_file = os.path.join(self.dataset_dir, self.dataset_name + ".txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            val = self.read_data(classnames, "val")
            test = self.read_data(classnames, "test")
            # test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "val": val, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames


    def read_data(self, classnames, split_dir):
        df_path = os.path.join(self.dataset_dir, self.dataset_name + '.csv')
        df = pd.read_csv(df_path, names=['set', 'name', 'label'])
        if 'train' in split_dir:
            img_list = df.loc[df['set'] == 'TRAIN']
        elif 'test' in split_dir:
            img_list = df.loc[df['set'] == 'TEST']
        else:
            img_list = df.loc[df['set'] == 'VALIDATION']

        img_list.reset_index()
        items = []

        for index, row in img_list.iterrows():
            impath = os.path.join(self.image_dir, row['name'])
            classname = classnames[str(row['label'])]
            item = Datum(impath=impath, label=row['label'], classname=classname)
            items.append(item)

        return items
