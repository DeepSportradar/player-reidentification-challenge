from __future__ import absolute_import, print_function

import os.path as osp

import numpy as np

from ..utils.data import Dataset
from ..utils.data.dataset import _pluck
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import read_json, write_json


class SynergyReID(Dataset):
    # md5 = "05715857791e2e88b2f11e4037fbec7d"
    md5 = "98fe8b85c154923742b7c14418f2b2d7"

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(SynergyReID, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it."
            )

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, "raw")
        mkdir_if_missing(raw_dir)

        # Open the raw zip file
        # fpath = osp.join(raw_dir, 'synergyreid_data.zip')
        fpath = osp.join(raw_dir, "data_reid.zip")

        if (
            osp.isfile(fpath)
            and hashlib.md5(open(fpath, "rb").read()).hexdigest() == self.md5
        ):
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please move data to {} ".format(fpath))

        # Extract the file
        exdir = osp.join(raw_dir, "data_reid")
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, "images")
        mkdir_if_missing(images_dir)

        # 487 identities (+1 for background) with 2 camera views each
        # Here we use the convention that camera 0 is for query and
        # camera 1 is for gallery
        identities = [[[] for _ in range(2)] for _ in range(487)]

        def register(subdir):
            fpaths = sorted(glob(osp.join(exdir, subdir, "*.jpeg")))
            pids = set()
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid = int(fname.split("_")[0])
                cam = 1 if "gallery" in subdir else 0
                pids.add(pid)
                fname = "{:08d}_{:02d}_{:04d}.jpg".format(
                    pid, cam, len(identities[pid][cam])
                )
                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
            return pids

        traintest_pids = register("reid_training")
        query_test_pids = register("reid_test/query")
        gallery_test_pids = register("reid_test/gallery")

        assert query_test_pids <= gallery_test_pids
        assert traintest_pids.isdisjoint(query_test_pids)

        # identities_challenge = [[[] for _ in range(2)] for _ in range(9172)]
        identities_challenge = [[[] for _ in range(2)] for _ in range(10128)]

        def register_challenge(subdir, n=0):
            fpaths = sorted(glob(osp.join(exdir, subdir, "*.jpeg")))
            pids = set()
            for pindx, fpath in enumerate(fpaths):
                fname = osp.basename(fpath)
                pid = int(fname.split(".")[0])
                cam = 1 if "gallery" in subdir else 0
                pids.add(pid)
                fname = "{:08d}_{:02d}_{:04d}.jpg".format(pid, cam, 0)
                try:
                    identities_challenge[pindx + n][cam].append(fname)
                except:
                    print(pindx, fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
            return pids

        query_challenge_pids = register_challenge("reid_challenge/query")
        gallery_challenge_pids = register_challenge(
            "reid_challenge/gallery", n=len(query_challenge_pids)
        )

        # Save the training / test / challenge splits
        splits = [
            {
                "traintest": sorted(list(traintest_pids)),
                "query_test": sorted(list(query_test_pids)),
                "gallery_test": sorted(list(gallery_test_pids)),
                "query_challenge": sorted(list(query_challenge_pids)),
                "gallery_challenge": sorted(list(gallery_challenge_pids)),
            }
        ]
        write_json(splits, osp.join(self.root, "splits.json"))

        # Save meta information into a json file
        meta = {
            "name": "SynergyReID",
            "shot": "multiple",
            "num_cameras": 2,
            "identities": identities,
            "identities_challenge": identities_challenge,
        }
        write_json(meta, osp.join(self.root, "meta.json"))

    def load(self, verbose=True):
        splits = read_json(osp.join(self.root, "splits.json"))
        if self.split_id >= len(splits):
            raise ValueError(
                "split_id exceeds total splits {}".format(len(splits))
            )
        self.split = splits[self.split_id]

        traintest_pids = np.concatenate(
            (
                np.asarray(self.split["traintest"]),
                np.asarray(self.split["query_test"]),
            )
        )

        def _pluck_test(identities, indices, relabel=False, cam=0):
            ret = []
            for index, pid in enumerate(indices):
                pid_images = identities[pid]
                for camid, cam_images in enumerate(pid_images):
                    if camid == cam:
                        for fname in cam_images:
                            name = osp.splitext(fname)[0]
                            x, y, _ = map(int, name.split("_"))
                            assert pid == x and camid == y
                            if relabel:
                                ret.append((fname, index, camid))
                            else:
                                ret.append((fname, pid, camid))
            return ret

        def _pluck_challenge(identities, indices, n=0):
            ret = []
            for index, pid in enumerate(indices):
                pid_images = identities[index + n]
                for camid, cam_images in enumerate(pid_images):
                    for fname in cam_images:
                        ret.append((fname, pid, camid))
            return ret

        self.meta = read_json(osp.join(self.root, "meta.json"))
        identities = self.meta["identities"]
        identities_challenge = self.meta["identities_challenge"]
        self.train = _pluck(identities, self.split["traintest"], relabel=True)
        self.traintest = _pluck(identities, traintest_pids, relabel=True)
        self.query_test = _pluck_test(
            identities, self.split["query_test"], cam=0
        )
        self.gallery_test = _pluck_test(
            identities, self.split["gallery_test"], cam=1
        )
        self.query_challenge = _pluck_challenge(
            identities_challenge, self.split["query_challenge"]
        )
        self.gallery_challenge = _pluck_challenge(
            identities_challenge,
            self.split["gallery_challenge"],
            n=len(self.split["query_challenge"]),
        )
        self.num_train_ids = len(self.split["traintest"])
        self.num_test_ids = len(self.split["query_test"])
        self.num_traintest_ids = len(traintest_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset      | # ids | # images")
            print("  ---------------------------")
            print(
                "  train       | {:5d} | {:8d}".format(
                    self.num_train_ids, len(self.train)
                )
            )
            print(
                "  query test   | {:5d} | {:8d}".format(
                    len(self.split["query_test"]), len(self.query_test)
                )
            )
            print(
                "  gallery test | {:5d} | {:8d}".format(
                    len(self.split["gallery_test"]), len(self.gallery_test)
                )
            )
            print(
                "  traintest    | {:5d} | {:8d}".format(
                    self.num_traintest_ids, len(self.traintest)
                )
            )
            print("  ---------------------------")
            print(
                "  query challenge  | {:5d} | {:8d}".format(
                    len(self.split["query_challenge"]),
                    len(self.query_challenge),
                )
            )
            print(
                "  gallery challenge | {:5d} | {:8d}".format(
                    len(self.split["gallery_challenge"]),
                    len(self.gallery_challenge),
                )
            )
