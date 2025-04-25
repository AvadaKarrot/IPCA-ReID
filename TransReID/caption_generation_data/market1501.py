from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings
from caption_generation_data.dataset import Dataset
class Market1501(Dataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'market'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)


        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = self.data_dir
        # data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        # caption试点
        self.caption = kwargs['caption'] if 'caption' in kwargs else False
        if self.caption: # 有caption的情况，需要在img_p, pid,camid后面加上caption
            train, query, gallery = self.process_caption(train, query, gallery)

        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(Market1501, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data

    def process_caption(self, train, query, gallery):
        # caption的数据结构：{img_path: caption}
        train_caption_path = osp.join(self.data_dir, 'train_caption.txt')
        query_caption_path = osp.join(self.data_dir, 'query_caption.txt')
        gallery_caption_path = osp.join(self.data_dir, 'gallery_caption.txt')
        train_caption, query_caption, gallery_caption = {}, {}, {}
        with open(train_caption_path, 'r') as f:
            for line in f:
                img_path, cap = line.strip().split()
                train_caption[img_path] = cap
        with open(query_caption_path, 'r') as f:
            for line in f:
                img_path, cap = line.strip().split()
                query_caption[img_path] = cap
        with open(gallery_caption_path, 'r') as f:
            for line in f:
                img_path, cap = line.strip().split()
                gallery_caption[img_path] = cap

        train = [(img_path, pid, camid, train_caption[img_path]) for img_path, pid, camid in train]
        query = [(img_path, pid, camid, query_caption[img_path]) for img_path, pid, camid in query]
        gallery = [(img_path, pid, camid, gallery_caption[img_path]) for img_path, pid, camid in gallery]

        return train, query, gallery
