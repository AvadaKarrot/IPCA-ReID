from __future__ import division, print_function, absolute_import
import os.path as osp

from ..dataset import ImageDataset

import json
# Log
# 22.01.2019
# - add v2
# - v1 and v2 differ in dir names
# - note that faces in v2 are blurred
TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'MSMT17_V1': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    }
}


class MSMT17(ImageDataset):
    """MSMT17.

    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_
    
    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    dataset_dir = 'msmt17'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        has_main_dir = False
        for main_dir in VERSION_DICT:
            if osp.exists(osp.join(self.dataset_dir, main_dir)):
                train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
                test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
                data_dir = VERSION_DICT[main_dir]
                has_main_dir = True
                break
        assert has_main_dir, 'Dataset folder not found'
        ## zwq
        self.data_dir = osp.join(self.dataset_dir, main_dir)                                                                
        ####
        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        self.list_train_path = osp.join(
            self.dataset_dir, main_dir, 'list_train.txt'
        )
        self.list_val_path = osp.join(
            self.dataset_dir, main_dir, 'list_val.txt'
        )
        self.list_query_path = osp.join(
            self.dataset_dir, main_dir, 'list_query.txt'
        )
        self.list_gallery_path = osp.join(
            self.dataset_dir, main_dir, 'list_gallery.txt'
        )

        required_files = [self.dataset_dir, self.train_dir, self.test_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, self.list_train_path)
        val = self.process_dir(self.train_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path)

        # caption试点
        self.caption = kwargs['caption'] if 'caption' in kwargs else False
        if self.caption: # 有caption的情况，需要在img_p, pid,camid后面加上caption
            self.cap_num = kwargs['cap_num'] if 'cap_num' in kwargs else False
            train, query, gallery = self.process_caption(train, query, gallery)


        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        if 'combineall' in kwargs and kwargs['combineall']:
            train += val

        super(MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            camid = int(img_path.split('_')[2]) - 1 # index starts from 0
            img_path = osp.join(dir_path, img_path)
            data.append((img_path, pid, camid))

        return data
############################################################ caption试点
    def process_caption(self, train, query, gallery):
        # caption的数据结构：{img_path: caption}
        train_caption_path = osp.join(self.data_dir, 'train_caption_dict.json')
        query_caption_path = osp.join(self.data_dir, 'query_caption_dict.json')
        gallery_caption_path = osp.join(self.data_dir, 'gallery_caption_dict.json')
        # test_caption_path = osp.join(self.data_dir, 'test_caption.txt')
        # train_caption, query_caption, gallery_caption, test_caption = {}, {}, {}, {}

        with open(train_caption_path, 'r') as f:
            train_caption = json.load(f)
        with open(query_caption_path, 'r') as f:
            query_caption = json.load(f)
        with open(gallery_caption_path, 'r') as f:
            gallery_caption = json.load(f)
        # with open(test_caption_path, 'r') as f:
        #     for line in f:
        #         img_path, cap = line.strip().split()
        #         test_caption[img_path] = cap

        def preprocess(captions):
            processed_captions = []
            for caption in captions:
                # 分割句子，并只保留第一个'.'前的部分
                first_sentence = caption.split('.')[0]
                processed_captions.append(first_sentence.strip().lower())
            
            return processed_captions

        for img_path, captions in train_caption.items():
            captions = preprocess(captions)
            selected_captions = [captions[i] for i in self.cap_num]  # 根据指定的索引选择元素
            train_caption[img_path] = ', '.join(selected_captions) # 用逗号分隔多个caption
        for img_path, captions in query_caption.items():
            captions = preprocess(captions)
            selected_captions = [captions[i] for i in self.cap_num]  # 根据指定的索引选择元素
            query_caption[img_path] = ', '.join(selected_captions) # 用逗号分隔多个caption   
        for img_path, captions in gallery_caption.items():
            captions = preprocess(captions)
            selected_captions = [captions[i] for i in self.cap_num]  # 根据指定的索引选择元素
            gallery_caption[img_path] = ', '.join(selected_captions) # 用逗号分隔多个caption      

        train = [(img_path, pid, camid, train_caption[img_path]) for img_path, pid, camid in train]
        query = [(img_path, pid, camid, query_caption[img_path]) for img_path, pid, camid in query]
        gallery = [(img_path, pid, camid, gallery_caption[img_path]) for img_path, pid, camid in gallery]

        return train, query, gallery