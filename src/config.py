import wandb
import os
import torch

def get_config():
    run = wandb.init(project='disentangling_minor_factors', entity='kadarsh22')
    wandb.run.name = 'test_run'
    wandb.run.save()

    config = wandb.config
    config.gan_type = 'pggan'  # choices=['ProgGAN', 'StyleGAN2','SNGAN','StyleGAN']
    config.dataset = 'CelebAHQ'  # choices=['AnimeFaceS',CelebAHQ' ,'LSUN-cars', 'LSUN-cats'
    config.model_name = 'pggan_celebahq1024'  # choices = ['pggan_celebahq1024',stylegan_animeface512,stylegan_car512,stylegan_cat256]
    config.initialisation = 'closed_form'
    config.random_seed = 123
    config.num_iterations = 10
    config.batch_size = 8
    config.deformator_type = 'ortho'  # choices = ['linear','ortho']
    config.deformator_lr = 0.0001
    config.eps_predictor_lr = 0.0001
    config.num_directions = 512
    config.latent_dim = 512
    config.epsilon = 10
    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config.train = True

    config.eval_samples = 8
    config.eval_eps = 10
    config.resume_direction = None

    config.saving_freq = 100
    config.logging_freq = 2
    config.evaluation_freq = 2
    config.visualisation_freq = 2

    config.classifier_path = 'pretrained_models'
    config.simple_cls_path = 'pretrained_models/classifiers/'
    config.nvidia_cls_path = 'pretrained_models/classifiers/nvidia_classifiers'
    config.result_path = os.path.join('results', wandb.run.name, 'qualitative_analysis')

    return config
