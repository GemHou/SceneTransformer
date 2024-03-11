import os
import sys
import hydra
import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn
from model.pl_module import SceneTransformer

import hydra
from hydra.core.config_store import ConfigStore

print("torch.cuda.is_available(): ", torch.cuda.is_available())


def get_config():
    # 初始化Hydra配置
    hydra.initialize(config_path='./conf')  # , config_name='config.yaml'

    # 获取配置变量
    cfg = hydra.compose(config_name='config.yaml')

    # 返回配置变量
    return cfg


# @hydra.main(config_path='./conf', config_name='config.yaml')
def main():  # cfg
    cfg = get_config()

    print("torch.cuda.is_available(): ", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_ids

    print("torch.cuda.is_available(): ", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise

    GPU_NUM = cfg.device_num
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(GPU_NUM))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM) / 1024 ** 3, 1), 'GB')

    pl.seed_everything(cfg.seed)
    # pwd = hydra.utils.get_original_cwd() + '/'
    pwd = "/home/houjinghp/study/SceneTransformer/"
    print('Current Path: ', pwd)

    dataset_train = WaymoDataset(pwd + cfg.dataset.train.tfrecords, pwd + cfg.dataset.train.idxs)
    dataset_valid = WaymoDataset(pwd + cfg.dataset.valid.tfrecords, pwd + cfg.dataset.valid.idxs)

    dloader_train = DataLoader(dataset_train, batch_size=cfg.dataset.batchsize, collate_fn=waymo_collate_fn)
    dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.batchsize, collate_fn=waymo_collate_fn)

    model = SceneTransformer(None, cfg.model.in_feature_dim, cfg.model.in_dynamic_rg_dim, cfg.model.in_static_rg_dim,
                             cfg.model.time_steps, cfg.model.feature_dim, cfg.model.head_num, cfg.model.k, cfg.model.F)

    print("prepare trainer...")
    trainer = pl.Trainer(max_epochs=cfg.max_epochs, gpus=1, auto_lr_find=True)

    # print("len(dloader_train.dataset): ", len(dloader_train.dataset))
    # from torchvision.datasets.utils import DatasetCatalog
    for data in tqdm.tqdm(dloader_train):
        # print("data: ", data)
        pass

    trainer.fit(model, dloader_train, dloader_valid)


if __name__ == '__main__':
    main()
    # sys.exit()
