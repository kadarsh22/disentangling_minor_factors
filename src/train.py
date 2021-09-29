import os.path
import random
from utils import *
import torch
import numpy as np
from models.attribute_predictors import attribute_utils, attribute_predictor
from torchvision import transforms
from models.latent_dataset import LatentDataset
from models import latent_regressor
import torchvision
import torch.nn.functional as F
import wandb


class Trainer(object):

    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.all_attr_list = ['Bald', 'Bangs', 'Goatee', 'Mustache' 'Pale_Skin',
                              'Wearing_Lipstick']

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def train_ours(self, generator, deformator, deformator_opt, eps_predictor, eps_predictor_opt, inversion_network):
        generator.zero_grad()
        deformator.zero_grad()
        eps_predictor_loss = 0
        deformator_ranking_loss = 0
        self.get_initialisations(generator)
        # deformator, direction_dict = self.initialise_directions(generator, deformator, inversion_network)

        return deformator, deformator_opt, eps_predictor, eps_predictor_opt, eps_predictor_loss, deformator_ranking_loss

    def train_inversion_network(self, generator, inversion_network):
        model = inversion_network.to(self.config.device)
        loader = self._get_encoder_train_data(generator)
        trained_model = latent_regressor._train(model, loader)
        return trained_model

    def initialise_directions(self, generator, deformator, inversion_network):
        supervision_images = [(1, 2), (3, 4), (5, 6), (6, 7), (7, 8), (7, 8), (7, 8)]
        inversion_network = self.train_inversion_network(generator, inversion_network)
        celeba_dataset = CelebADataset(self.config.image_path, transforms.Compose([transforms.ToTensor()]))
        direction_dict = {}
        for idx, (positive, negative) in enumerate(supervision_images):
            positive_dir = inversion_network(celeba_dataset.__getitem__(positive))
            negative_dir = inversion_network(celeba_dataset.__getitem__(negative))
            direction_attr = positive_dir - negative_dir
            deformator.ortho_mat.data[:, idx] = direction_attr
            direction_dict[self.all_attr_list[idx]] = direction_attr
        return deformator, direction_dict

    def get_initialisations(self ,generator):
        # celeba_dataset = CelebADataset(self.config.image_path, transforms.Compose([transforms.Resize(256),transforms.ToTensor()]))
        # pool_loader = torch.utils.data.DataLoader(celeba_dataset, batch_size=self.config.batch_size, num_workers=0,
        #                                           pin_memory=True, shuffle=False, drop_last=True)
        z = torch.load('results/best_z_corresponding_scores/z_generated.pth')
        ordered_idx = torch.load('results/best_z_corresponding_scores/ordered_idx.pth')
        classifier_name = 'Bald'

        z_full = torch.randn(30000, self.config.latent_dim)
        os.makedirs(os.path.join(self.config.result_path, "generated_images"), exist_ok=True)
        torch.save(z_full,os.path.join(self.config.result_path, "generated_images", "z_generated.pth"))
        new_dataset = NoiseDataset(z_full)
        z_loader = torch.utils.data.DataLoader(new_dataset, batch_size=self.config.batch_size,
                                                             num_workers=0,
                                                             pin_memory=True, shuffle=False, drop_last=True)
        initialisation_artifact = wandb.Artifact(str(wandb.run.name) + 'initialisation', type="initialisations")
        extreme_ = wandb.Table(columns=['image_grid', 'direction_idx'])

        ordered_idx = {}
        classifier_list = []
        for predictor_idx, classifier_name in enumerate(self.all_attr_list):
            predictor = attribute_utils.ClassifierWrapper(classifier_name, ckpt_path=self.config.nvidia_cls_path,
                                                          device=self.config.device)
            predictor.to(self.config.device).eval()
            classifier_scores = []
            for batch_idx, z in enumerate(z_loader):
                images = torch.clamp(F.avg_pool2d(generator(z), 4, 4), min=-1, max=1)
                scores = torch.softmax(predictor(images.to(self.config.device)), dim=1)[:, 1]
                classifier_scores = classifier_scores + scores.detach().tolist()
            print(len(classifier_scores))
            classifier_scores_array = np.array(classifier_scores)
            ordered_idx[str(classifier_name)] = classifier_scores_array
            smallest_idx = classifier_scores_array.argsort()[:10]
            largest_idx = classifier_scores_array.argsort()[-10:][::-1]
            print(classifier_name)
            print("-------smallest_idx-------")
            print(smallest_idx)
            print("-------largest_idx --------")
            print(largest_idx)
            indx = smallest_idx.tolist() + largest_idx.tolist()
            image_array = torch.stack([torch.clamp(F.avg_pool2d(generator(z_full[idx].view(-1,self.config.latent_dim)), 4, 4), min=-1, max=1) for idx in indx])
            grid = torchvision.utils.make_grid(image_array, nrow=10, scale_each=True, normalize=True)
            extreme_.add_data(wandb.Image(grid), self.all_attr_list[predictor_idx])
        os.makedirs(os.path.join(self.config.result_path, "sorted_images"), exist_ok=True)
        torch.save(ordered_idx, os.path.join(self.config.result_path, "sorted_images", "ordered_idx.pth"))
        initialisation_artifact.add(extreme_, "initialisations")
        wandb.run.log_artifact(initialisation_artifact, aliases=str(0))

    def _get_encoder_train_data(self, generator):
        save_dir = os.path.join(self.config.result_path, 'generated_data')
        os.makedirs(save_dir, exist_ok=True)
        train_dataset = LatentDataset(generator, save_dir, create_new_data=True)

        LABEL_MEAN = np.mean(train_dataset.labels, 0)
        LABEL_STD = np.std(train_dataset.labels, 0) + 1e-5

        train_dataset.labels = (train_dataset.labels - LABEL_MEAN) / LABEL_STD

        test_dataset = LatentDataset(generator, save_dir, create_new_data=True)

        test_dataset.labels = (test_dataset.labels - LABEL_MEAN) / LABEL_STD

        val_dataset = LatentDataset(generator, save_dir, create_new_data=True)

        val_dataset.labels = (val_dataset.labels - LABEL_MEAN) / LABEL_STD

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.encoder_batch_size,
                                                   pin_memory=True, shuffle=True)

        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.encoder_batch_size,
                                                   pin_memory=True, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.encoder_batch_size,
                                                  shuffle=False)

        return {"train": train_loader, "valid": valid_loader, "test": test_loader}
