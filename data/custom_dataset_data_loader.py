import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


def CreateDataset_dataprocess(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset_dataprocess()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


def CreateDataset_Real(opt):
    dataset_real = None
    from data.aligned_dataset import AlignedDataset_Real
    dataset_real = AlignedDataset_Real()

    print("dataset [%s] was created" % (dataset_real.name()))
    dataset_real.initialize(opt)
    return dataset_real


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)

        self.real_train = opt.real_train if opt.isTrain else False
        
        if self.real_train:
            self.dataset = CreateDataset(opt)
            self.dataset_real = CreateDataset_Real(opt)
            
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads))

            self.dataloader_real = torch.utils.data.DataLoader(
                self.dataset_real,
                batch_size=opt.real_batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads))
        else:
            self.dataset = CreateDataset(opt)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads))

    def load_data(self):
        if self.real_train:
            return self.dataloader,self.dataloader_real        
        else:
            return self.dataloader

    def __len__(self):
        return len(self.dataset)





# def CreateDataset(opt):
#     dataset = None
#     from data.aligned_dataset import AlignedDataset
#     dataset = AlignedDataset()

#     print("dataset [%s] was created" % (dataset.name()))
#     dataset.initialize(opt)
#     return dataset

# class CustomDatasetDataLoader(BaseDataLoader):
#     def name(self):
#         return 'CustomDatasetDataLoader'

#     def initialize(self, opt):
#         BaseDataLoader.initialize(self, opt)
#         self.dataset = CreateDataset(opt)
#         self.dataloader = torch.utils.data.DataLoader(
#             self.dataset,
#             batch_size=opt.batchSize,
#             shuffle=not opt.serial_batches,
#             num_workers=int(opt.nThreads))

#     def load_data(self):
#         return self.dataloader

#     def __len__(self):
#         return min(len(self.dataset), self.opt.max_dataset_size)