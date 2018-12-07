from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data.sampler import Sampler
import itertools
def make_dataloader(args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize]
    )
       
    trainset = CIFAR10(args, root=args.dir_data, train=True, semi_supervised=args.semi_supervised,
                        download=True, transform = transform)

        
    if not args.semi_supervised :
        sampler = SubsetRandomSampler(trainset.labeled)
        batch_sampler = BatchSampler(sampler, 
                                    args.batch_size, 
                                    drop_last=True)
    else:
        #labeled_sampler = SubsetRandomSampler(trainset.labeled)
        #unlabeled_sampler = SubsetRandomSampler(trainset.unlabeled)
        
        batch_sampler = TwoBatchSampler(trainset.labeled, trainset.unlabeled,
                                    args.batch_size,
                                    args.labeled_batch_size, drop_last=True)


    """  
    trainset = torchvision.datasets.CIFAR10(root=args.dir_data, train=True,
                                            download=True, transform=transform)
    """
    trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler,
                                            num_workers = args.num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root=args.dir_data, train=False,
                                            download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    testloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size,
                                            shuffle = False, num_workers = args.num_workers, pin_memory=True,
                                            drop_last = False)

    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

def prepare_dataset(X_train, y_train, num_labeled):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    num_classes = 10
    num_img = min(num_classes, 20)
    max_count = num_labeled // num_classes
    print("Keeping %d labels per 10 classes." % max_count)
    #label_image = np.zeros((num_img * max_count, X_train.shape[1], 32, 3))
    mask_train = np.zeros(len(y_train), dtype=np.float32)
    count = [0] * num_classes
    labeled = []
    unlabeled = []
    for i in range(len(y_train)):
        idx = indices[i]
        label = y_train[idx]
        if count[label] < max_count:
            count[label] += 1
            labeled.append(idx)
        else:
            unlabeled.append(idx)
            #y_train[idx] = 11
    """
    for i in range(len(y_train)):
        if mask_train[i] != 1.0:
            y_train[i] = 0
    """

    return labeled, unlabeled

class CIFAR10(data.Dataset):

    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, args, root, train=True, semi_supervised=False,
                 transform=None, target_transform=None,
                 download=False):
        self.args = args
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.semi_supervised = semi_supervised
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')


        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        if self.train:
            self.labeled, self.unlabeled = prepare_dataset(self.data, self.targets, self.args.labeled)
            """
            if not self.semi_supervised and self.args.labeled < 50000:
                self.data = self.data[self.labeled]
                self.targets = np.asarray(self.targets)
                self.targets = self.targets[self.unlabeled].tolist()
            """    

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)

            if not check_integrity(fpath, md5):
                return False

        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    

class TwoBatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, labeled_indices, unlabeled_indices, batch_size, labeled_batch_size, drop_last):
        """
        if not isinstance(labeled_sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(labeled_sampler))
        if not isinstance(unlabeled_sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(unlabeled_sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(labeled_batch_size, int) or isinstance(labeled_batch_size, bool) or \
                labeled_batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        """
        self.labeled_indices = labeled_indices
        self.unlabeled_indices= unlabeled_indices
        self.batch_size = batch_size
        self.labeled_batch_size = labeled_batch_size
        self.drop_last = drop_last

    def __iter__(self):
        labeled_indices = iterate_eternally(self.labeled_indices)
        unlabeled_indices = iterate_once(self.unlabeled_indices)
        batch = []
        for idx in labeled_indices:
            batch.append(idx)
            if len(batch) == self.labeled_batch_size:
                break
        batch2 = []
        for idx in unlabeled_indices:
            batch2.append(idx)
            if len(batch2) == self.batch_size - self.labeled_batch_size:
                break
        
        yield batch + batch2


    def __len__(self):
        return len(self.unlabeled_indices) // (self.batch_size - self.labeled_batch_size)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())
