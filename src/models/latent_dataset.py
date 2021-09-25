import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import random
import logging


class LatentDataset(Dataset):
    def __init__(self, generator, config, seed, create_new_data=False):
        super().__init__()
        assert config.encoder.num_samples % config.encoder_batch_size == 0
        self.config = config
        self.set_seed(seed)

        if create_new_data:
            self._generate_data(generator, self.config.encoder_batch_size, self.config.dataset,
                                N=self.config.encoder_num_samples, save=False)
        else:
            exist = self._try_load_cached(self.config.dataset)
            if not exist:
                print("Building dataset from scratch.")
                self._generate_data(generator=generator, generator_bs=self.config.encoder_batch_size,
                                    dataset=self.config.dataset,
                                    N=self.config.encoder.num_samples, save=True)

    def _try_load_cached(self, dataset):
        path = os.path.join(self.config.result_path, dataset + ".npz")
        if os.path.exists(path):
            "Loading cached dataset."
            arr = np.load(path)
            self.images, self.labels = arr["images"], arr["labels"]
            return True
        else:
            return False

    @torch.no_grad()
    def _generate_data(self, generator, generator_bs, dataset, N, save=True):
        images = []
        labels = []

        for _ in range(N // generator_bs):
            z = torch.randn(generator_bs, generator.style_dim).to(self.config.device) ##TODO convert from style gan to
            w = generator.style(z)
            x = generator([w])[0]
            x = torch.clamp(x, -1, 1)
            x = (((x.detach().cpu().numpy() + 1) / 2) * 255).astype(np.uint8)
            images.append(x)
            labels.append(w.detach().cpu().numpy())

        self.images = np.concatenate(images, 0)
        logging.info('max value in image :' + str(self.images.max()))
        self.labels = np.concatenate(labels, 0)
        if save:
            path = os.path.join(self.config.result_path, dataset + ".npz")
            np.savez(path, images=self.images, labels=self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]
        img = 2 * (img / 255) - 1
        return torch.from_numpy(img).float(), torch.from_numpy(self.labels[item]).float()

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)