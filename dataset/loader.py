import os
import sqlite3
import sys
from dataclasses import dataclass
from glob import glob
from typing import Any, Optional

import h5py
import torchvision.transforms as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import random
import math
import pickle
import torch

VALID_IMAGE_EXTENSIONS = [
    "bmp",
    "gif",
    "jfif",
    "jpeg",
    "jpg",
    "pgm",
    "png",
    "ppm",
    "tif",
    "webp",
]



SET_TYPES = ["train", "validation", "test"]
INPUT_TYPES = ["original", "features"]
VECTOR_TYPES = ["resnet50", "xception"]

# given by Yuecheng
resnet50_tf_pipeline = TF.Compose(
    [
        TF.Resize((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# resize, crop, and normalize (matches pipeline for xception in timm)
xception_tf_pipeline = TF.Compose(
    [
        TF.Resize(
            size=333,
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias=None,
        ),
        TF.CenterCrop(size=(299, 299)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


@dataclass
class CollectionImage:
    # data required for every image
    relative_path: str
    class_id: int
    set_id: int  # named to prevent issues in sqlite calls
    problem: int

    # since subject_id is rare, default to not requiring it
    # subject_id: int = -1

    # data that is calculated later
    # file_size: int = -1
    file_hash: str = ""
    # image_hash: str = ""
    # image_width: int = -1
    # image_height: int = -1
    # image_mode: str = ""
    # image_format: str = ""

    # feature vector
    feature_vector: Tensor = Tensor([])


class CollectionDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        set_type: str,
        input_type: str,
        vector_type: Optional[str] = "resnet50",
        pipeline: Optional[Any] = resnet50_tf_pipeline,
        label_dict = None,
        verify_all_sets_all_types: bool = False,
        full_dataset = False, 
        root_path = "/lab/tmpig15b/u/"
    ) -> None:
        super().__init__()
        """init function for the dataset"""
        random.seed(27)

        self.initialized_correctly: bool = False
        self.images_root_path = root_path + "name-pending_collection/"
        self.vectors_root_path = root_path + "name-pending_collection_vectors/"
        self.database_path = root_path + "name-pending_collection/0_collection/all_images.sqlite"

        # check inputs
        root_path = CollectionDataset.sanitize_and_check_root_path(self.images_root_path)
        assert len(dataset_name) != ""
        assert set_type in SET_TYPES
        assert input_type in INPUT_TYPES

        if input_type == "features":
            assert vector_type in VECTOR_TYPES

        # store inputs
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.set_type = set_type
        self.input_type = input_type
        self.vector_type = vector_type
        self.pipeline = pipeline

        # get the dataset_name
        # query = f"select dataset_name FROM datasets where dataset={dataset};"
        # result = CollectionDataset.execute_database_query(root_path, query)
        # if len(result) == 0:
        #     print("no dataset with that id")
        #     return None

        # self.dataset_name = result[0][0]

        # grab the correct rows
        print("load relevant results from database...")
        set_type_int = SET_TYPES.index(self.set_type)
        if not verify_all_sets_all_types:
            query = f'SELECT file_hash, relative_path, class_id, set_id, problem FROM images WHERE dataset="{dataset_name}" AND problem=0 AND set_id={set_type_int};'
        else:
            query = f'SELECT file_hash, relative_path, class_id, set_id, problem FROM images WHERE dataset="{dataset_name}" AND problem=0;'
        rows = CollectionDataset.execute_database_query(query, self.database_path)

        if len(rows) == 0:
            print(
                f"no results for dataset_name={dataset_name} and set_type={set_type_int}"
            )
            return None

        self.images: list[CollectionImage] = []
        self.dict = {}
        self.label_dict = {}

        if full_dataset:
            self.sample_class_num = math.inf
            self.sample_train_size = math.inf
            self.sample_test_size = math.inf
        else:
            self.sample_class_num = 500
            self.sample_train_size = 54000
            self.sample_test_size = 6000
        for row in rows:
            if int(row[2]) not in self.dict:
                self.dict[int(row[2])] = [(row[0], row[1], row[3], row[4])]
            else:
                self.dict[int(row[2])].append((row[0], row[1], row[3], row[4]))
        
        if label_dict:
            self.label_dict = label_dict
        else:
            """
            modification here: first tries to implement an identity mapping if possible
            """
            if set(self.dict.keys()) == set(list(range(len(self.dict.keys())))):
                self.label_dict = {i:i for i in range(len(self.dict.keys()))}
            else:
                label_use = 0
                for label in self.dict.keys():
                    self.label_dict[label] = label_use
                    label_use += 1

        if len(self.dict.keys()) > self.sample_class_num:
            self.num_classes = self.sample_class_num
        else:
            self.num_classes = len(self.dict.keys())
            
        if self.set_type=="train" and len(rows) > self.sample_train_size:
            self.labels, self.hashes, self.paths, self.id_set, self.problem_list = self._random_value(self.dict, self.sample_train_size)
        elif (self.set_type == "validation" or self.set_type == "test") and len(rows) > self.sample_test_size:
            self.labels, self.hashes, self.paths, self.id_set, self.problem_list = self._random_value(self.dict, self.sample_test_size)
        else:
            self.labels, self.hashes, self.paths, self.id_set, self.problem_list = self._random_value(self.dict, math.inf)
        for i in range(len(self.labels)):
            self.labels[i] = self.label_dict[self.labels[i]]
            self.images.append(CollectionImage(file_hash=self.hashes[i], 
                                                relative_path=self.paths[i], 
                                                class_id=self.labels[i], 
                                                set_id=int(self.id_set[i]),
                                                problem=int(self.problem_list[i]),
                                                ))

        # store the details for the images
        # for row in rows:
        #     self.images += [
        #         CollectionImage(
        #             file_hash=row[0],
        #             relative_path=row[1],
        #             class_id=int(row[2]),
        #             set_id=int(row[3]),
        #             problem=int(row[4]),
        #         )
        #     ]

        if not verify_all_sets_all_types:
            if self.input_type == "original":
                self.initialized_correctly = self.verify_original_images_exist()
            else:  # features input
                self.initialized_correctly = (
                    self.verify_feature_vectors_exist_and_load()
                )
        else:
            # assume true, but set to false if any fail
            self.initialized_correctly = True
            if not self.verify_original_images_exist():
                self.initialized_correctly = False
            if not self.verify_feature_vectors_exist_and_load(skip_load=True):
                self.initialized_correctly = False

        if self.initialized_correctly:
            print("done")
        else:
            print("initialization failed")

    def _random_value(self, dictionary, amount):
        label = []
        hash = []
        path = []
        set_id = []
        problem_id = []
        if amount == math.inf:
            num_each_class = math.inf
        else:
            num_each_class = round(amount/self.num_classes)

        current_class_size = 0
        for key, values in dictionary.items():
            track_num = 0
            if key not in list(self.label_dict.keys())[:self.num_classes]:
                assert self.num_classes == self.sample_class_num
                if self.num_classes == self.sample_class_num:
                    continue
            random.shuffle(values)
            for value in values:
                label.append(key)
                hash.append(value[0])
                path.append(value[1])
                set_id.append(value[2])
                problem_id.append(value[3])
                track_num += 1
                if track_num >= num_each_class:
                    break
            current_class_size += 1
            if current_class_size >= self.num_classes:
                break
        return label, hash, path, set_id, problem_id


    def __len__(self) -> int:
        assert self.initialized_correctly

        return len(self.images)

    def __getitem__(self, index: int) -> tuple:
        assert self.initialized_correctly
        assert 0 <= index <= len(self.images)

        if self.input_type == "features":
            return torch.from_numpy(self.images[index].feature_vector), self.images[index].class_id
        else:

            image_filename = self.original_image_path_for_index(index)

            with Image.open(image_filename) as image:

                # for the original images, make sure to convert to RGB
                if self.input_type == "original":
                    image = image.convert("RGB")

                # convert to a tensor before doing anything else
                # image_tensor = functional.to_tensor(image)

            # if there is a pipeline, run it
            if self.pipeline:
                image_tensor = self.pipeline(image)

            return image_tensor, self.images[index].class_id

    def original_image_path_for_index(self, index: int) -> str:
        """get the path to the original image"""
        assert 0 <= index <= len(self.images)

        return (
            self.root_path + self.dataset_name + "/" + self.images[index].relative_path
        )

    def verify_original_images_exist(self) -> bool:
        """check that all original image files exist"""

        had_error = False
        print("verifying that original images exist...")

        # order is not important
        for index in tqdm(range(len(self.images))):
            original_filename = self.original_image_path_for_index(index)
            if not os.path.isfile(original_filename):
                print("missing", original_filename)
                had_error = True

        return not had_error

    def verify_feature_vectors_exist_and_load(self, skip_load: bool = False) -> bool:
        """check that the hdf5 file exists, and load all of the feature vectors for the set"""

        if self.vector_type is None:
            print("error with vector_type")
            return False

        had_error = True
        feature_vectors_filename = (
            self.vectors_root_path + self.vector_type + "/" + self.dataset_name + ".h5"
        )

        if not skip_load:
            print(
                f"verifying all {self.vector_type} feature vectors exist, and loading..."
            )
        else:
            print(f"verifying all {self.vector_type} feature vectors exist...")

        if not os.path.isfile(feature_vectors_filename):
            print("feature vector file is missing", feature_vectors_filename)
        else:

            all_hashes = []
            # force order to be the same
            for i in range(len(self.images)):
                all_hashes += [self.images[i].file_hash]

            with h5py.File(feature_vectors_filename) as file_h:
                # check that all hashes exist in file
                feature_vectors_hashes = list(file_h.keys())
                missing_list = list(set(all_hashes) - set(feature_vectors_hashes))

                if len(missing_list) > 0:
                    print("missing these feature vectors:", missing_list)
                else:
                    # if no issues, create matrix of feature vectors
                    if not skip_load:
                        for i, _ in enumerate(self.images):
                            self.images[i].feature_vector = file_h[self.images[i].file_hash][:]  # type: ignore
                    had_error = False

        return not had_error

    def get_entry_for_index(self, index: int) -> CollectionImage:
        """get the entry for a given index"""
        assert self.initialized_correctly
        assert 0 <= index <= len(self.images)

        return self.images[index]

    @property
    def unique_class_id_list(self) -> list:
        """get a list of unique class_id"""
        assert len(self.images) > 0

        all_class_ids = [x.class_id for x in self.images]
        # return an ordered and unique list
        return sorted(list(set(all_class_ids)))

    @property
    def class_id_counts(self) -> dict:
        """get the counts for all class_ids"""
        assert len(self.images) > 0

        counts = {}
        all_class_ids = [x.class_id for x in self.images]
        for class_id in self.unique_class_id_list:
            counts.update({class_id: all_class_ids.count(class_id)})

        return counts

    @staticmethod
    def test_if_valid_image(full_file_path: str) -> bool:
        """function to test if an image file is valid
        (this will catch most, but not 100% of all image issues)"""
        try:
            with Image.open(full_file_path) as image_h:
                try:
                    image_h.verify()
                    return True
                except:
                    # failed after the file opened, while trying to verify
                    return False
        except:
            # failed in trying to open the image file
            return False

    @staticmethod
    def sanitize_and_check_root_path(root_path: str) -> str:
        """make sure root path ends in '/', and that it exists"""
        root_path = (root_path + "/").replace("//", "/")
        assert os.path.isdir(root_path)

        return root_path

    @staticmethod
    def execute_database_query(query: str, database_path) -> list:
        """execute a query on the database at the path"""

        # root_path = CollectionDataset.sanitize_and_check_root_path(root_path)
        database_filename = database_path  # root_path + "database.sqlite"

        assert os.path.isfile(database_filename)

        with sqlite3.connect(database_filename) as sql_conn:
            cursor = sql_conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

        return rows


if __name__ == "__main__":
    pass