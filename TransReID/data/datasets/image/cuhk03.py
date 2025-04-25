from __future__ import division, print_function, absolute_import
import os.path as osp

from torchreid.utils import read_json, write_json, mkdir_if_missing

from ..dataset import ImageDataset
import json

class CUHK03(ImageDataset):
    """CUHK03.

    Reference:
        Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!>`_
    
    Dataset statistics:
        - identities: 1360.
        - images: 13164.
        - cameras: 6.
        - splits: 20 (classic).
    """
    dataset_dir = 'cuhk03'
    dataset_url = None

    def __init__(
        self,
        root='',
        split_id=0,
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
        **kwargs
    ):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.data_dir = osp.join(self.dataset_dir, 'cuhk03_release')
        self.raw_mat_path = osp.join(self.data_dir, 'cuhk-03.mat')

        self.imgs_detected_dir = osp.join(self.dataset_dir, 'images_detected')
        self.imgs_labeled_dir = osp.join(self.dataset_dir, 'images_labeled')

        self.split_classic_det_json_path = osp.join(
            self.dataset_dir, 'splits_classic_detected.json'
        )
        self.split_classic_lab_json_path = osp.join(
            self.dataset_dir, 'splits_classic_labeled.json'
        )

        self.split_new_det_json_path = osp.join(
            self.dataset_dir, 'splits_new_detected.json'
        )
        self.split_new_lab_json_path = osp.join(
            self.dataset_dir, 'splits_new_labeled.json'
        )

        self.split_new_det_mat_path = osp.join(
            self.dataset_dir, 'cuhk03_new_protocol_config_detected.mat'
        )
        self.split_new_lab_mat_path = osp.join(
            self.dataset_dir, 'cuhk03_new_protocol_config_labeled.mat'
        )

        required_files = [
            self.dataset_dir, self.data_dir, self.raw_mat_path,
            self.split_new_det_mat_path, self.split_new_lab_mat_path
        ]
        self.check_before_run(required_files)

        self.preprocess_split()

        if cuhk03_labeled:
            split_path = self.split_classic_lab_json_path if cuhk03_classic_split else self.split_new_lab_json_path
        else:
            split_path = self.split_classic_det_json_path if cuhk03_classic_split else self.split_new_det_json_path

        splits = read_json(split_path)
        assert split_id < len(
            splits
        ), 'Condition split_id ({}) < len(splits) ({}) is false'.format(
            split_id, len(splits)
        )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        # caption试点
        self.caption = kwargs['caption'] if 'caption' in kwargs else False
        if self.caption: # 有caption的情况，需要在img_p, pid,camid后面加上caption
            self.cap_num = kwargs['cap_num'] if 'cap_num' in kwargs else False
            train, query, gallery = self.process_caption(train, query, gallery)

        super(CUHK03, self).__init__(train, query, gallery, **kwargs)

    def preprocess_split(self):
        # This function is a bit complex and ugly, what it does is
        # 1. extract data from cuhk-03.mat and save as png images
        # 2. create 20 classic splits (Li et al. CVPR'14)
        # 3. create new split (Zhong et al. CVPR'17)
        if osp.exists(self.imgs_labeled_dir) \
           and osp.exists(self.imgs_detected_dir) \
           and osp.exists(self.split_classic_det_json_path) \
           and osp.exists(self.split_classic_lab_json_path) \
           and osp.exists(self.split_new_det_json_path) \
           and osp.exists(self.split_new_lab_json_path):
            return

        import h5py
        import imageio
        from scipy.io import loadmat

        mkdir_if_missing(self.imgs_detected_dir)
        mkdir_if_missing(self.imgs_labeled_dir)

        print(
            'Extract image data from "{}" and save as png'.format(
                self.raw_mat_path
            )
        )
        mat = h5py.File(self.raw_mat_path, 'r')

        def _deref(ref):
            return mat[ref][:].T

        def _process_images(img_refs, campid, pid, save_dir):
            img_paths = [] # Note: some persons only have images for one view
            for imgid, img_ref in enumerate(img_refs):
                img = _deref(img_ref)
                if img.size == 0 or img.ndim < 3:
                    continue # skip empty cell
                # images are saved with the following format, index-1 (ensure uniqueness)
                # campid: index of camera pair (1-5)
                # pid: index of person in 'campid'-th camera pair
                # viewid: index of view, {1, 2}
                # imgid: index of image, (1-10)
                viewid = 1 if imgid < 5 else 2
                img_name = '{:01d}_{:03d}_{:01d}_{:02d}.png'.format(
                    campid + 1, pid + 1, viewid, imgid + 1
                )
                img_path = osp.join(save_dir, img_name)
                if not osp.isfile(img_path):
                    imageio.imwrite(img_path, img)
                img_paths.append(img_path)
            return img_paths

        def _extract_img(image_type):
            print('Processing {} images ...'.format(image_type))
            meta_data = []
            imgs_dir = self.imgs_detected_dir if image_type == 'detected' else self.imgs_labeled_dir
            for campid, camp_ref in enumerate(mat[image_type][0]):
                camp = _deref(camp_ref)
                num_pids = camp.shape[0]
                for pid in range(num_pids):
                    img_paths = _process_images(
                        camp[pid, :], campid, pid, imgs_dir
                    )
                    assert len(img_paths) > 0, \
                        'campid{}-pid{} has no images'.format(campid, pid)
                    meta_data.append((campid + 1, pid + 1, img_paths))
                print(
                    '- done camera pair {} with {} identities'.format(
                        campid + 1, num_pids
                    )
                )
            return meta_data

        meta_detected = _extract_img('detected')
        meta_labeled = _extract_img('labeled')

        def _extract_classic_split(meta_data, test_split):
            train, test = [], []
            num_train_pids, num_test_pids = 0, 0
            num_train_imgs, num_test_imgs = 0, 0
            for i, (campid, pid, img_paths) in enumerate(meta_data):

                if [campid, pid] in test_split:
                    for img_path in img_paths:
                        camid = int(
                            osp.basename(img_path).split('_')[2]
                        ) - 1 # make it 0-based
                        test.append((img_path, num_test_pids, camid))
                    num_test_pids += 1
                    num_test_imgs += len(img_paths)
                else:
                    for img_path in img_paths:
                        camid = int(
                            osp.basename(img_path).split('_')[2]
                        ) - 1 # make it 0-based
                        train.append((img_path, num_train_pids, camid))
                    num_train_pids += 1
                    num_train_imgs += len(img_paths)
            return train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs

        print('Creating classic splits (# = 20) ...')
        splits_classic_det, splits_classic_lab = [], []
        for split_ref in mat['testsets'][0]:
            test_split = _deref(split_ref).tolist()

            # create split for detected images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_detected, test_split)
            splits_classic_det.append(
                {
                    'train': train,
                    'query': test,
                    'gallery': test,
                    'num_train_pids': num_train_pids,
                    'num_train_imgs': num_train_imgs,
                    'num_query_pids': num_test_pids,
                    'num_query_imgs': num_test_imgs,
                    'num_gallery_pids': num_test_pids,
                    'num_gallery_imgs': num_test_imgs
                }
            )

            # create split for labeled images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_labeled, test_split)
            splits_classic_lab.append(
                {
                    'train': train,
                    'query': test,
                    'gallery': test,
                    'num_train_pids': num_train_pids,
                    'num_train_imgs': num_train_imgs,
                    'num_query_pids': num_test_pids,
                    'num_query_imgs': num_test_imgs,
                    'num_gallery_pids': num_test_pids,
                    'num_gallery_imgs': num_test_imgs
                }
            )

        write_json(splits_classic_det, self.split_classic_det_json_path)
        write_json(splits_classic_lab, self.split_classic_lab_json_path)

        def _extract_set(filelist, pids, pid2label, idxs, img_dir, relabel):
            tmp_set = []
            unique_pids = set()
            for idx in idxs:
                img_name = filelist[idx][0]
                camid = int(img_name.split('_')[2]) - 1 # make it 0-based
                pid = pids[idx]
                if relabel:
                    pid = pid2label[pid]
                img_path = osp.join(img_dir, img_name)
                tmp_set.append((img_path, int(pid), camid))
                unique_pids.add(pid)
            return tmp_set, len(unique_pids), len(idxs)

        def _extract_new_split(split_dict, img_dir):
            train_idxs = split_dict['train_idx'].flatten() - 1 # index-0
            pids = split_dict['labels'].flatten()
            train_pids = set(pids[train_idxs])
            pid2label = {pid: label for label, pid in enumerate(train_pids)}
            query_idxs = split_dict['query_idx'].flatten() - 1
            gallery_idxs = split_dict['gallery_idx'].flatten() - 1
            filelist = split_dict['filelist'].flatten()
            train_info = _extract_set(
                filelist, pids, pid2label, train_idxs, img_dir, relabel=True
            )
            query_info = _extract_set(
                filelist, pids, pid2label, query_idxs, img_dir, relabel=False
            )
            gallery_info = _extract_set(
                filelist,
                pids,
                pid2label,
                gallery_idxs,
                img_dir,
                relabel=False
            )
            return train_info, query_info, gallery_info

        print('Creating new split for detected images (767/700) ...')
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_det_mat_path), self.imgs_detected_dir
        )
        split = [
            {
                'train': train_info[0],
                'query': query_info[0],
                'gallery': gallery_info[0],
                'num_train_pids': train_info[1],
                'num_train_imgs': train_info[2],
                'num_query_pids': query_info[1],
                'num_query_imgs': query_info[2],
                'num_gallery_pids': gallery_info[1],
                'num_gallery_imgs': gallery_info[2]
            }
        ]
        write_json(split, self.split_new_det_json_path)

        print('Creating new split for labeled images (767/700) ...')
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_lab_mat_path), self.imgs_labeled_dir
        )
        split = [
            {
                'train': train_info[0],
                'query': query_info[0],
                'gallery': gallery_info[0],
                'num_train_pids': train_info[1],
                'num_train_imgs': train_info[2],
                'num_query_pids': query_info[1],
                'num_query_imgs': query_info[2],
                'num_gallery_pids': gallery_info[1],
                'num_gallery_imgs': gallery_info[2]
            }
        ]
        write_json(split, self.split_new_lab_json_path)
################ caption try

    def process_caption(self, train, query, gallery):
        # caption的数据结构：{img_path: caption}
        train_caption_path = osp.join(self.dataset_dir, 'train_caption_dict.json')
        query_caption_path = osp.join(self.dataset_dir, 'query_caption_dict.json')
        gallery_caption_path = osp.join(self.dataset_dir, 'gallery_caption_dict.json')
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