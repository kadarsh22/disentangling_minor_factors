import wandb
import os
import torch


def get_config(args):
    wandb.init(project='disentangling_minor_factors', entity='kadarsh22')
    wandb.run.name = args['exp_name']
    wandb.run.notes = args["exp_desc"]
    wandb.run.save()

    config = wandb.config
    config.gan_type = 'StyleGAN'  # choices=['pggan', 'StyleGAN2','SNGAN','StyleGAN']
    config.dataset = 'LSUN-cats'  # choices=['AnimeFaceS',CelebAHQ' ,'LSUN-cars', 'LSUN-cats']
    config.source_model_name = 'stylegan_cat256'  # choices = ['pggan_celebahq1024',stylegan_animeface512, # stylegan_car512,stylegan_cat256]
    config.target_model_name = 'pggan_celebahq1024'

    config.initialisation = 'closed_form'
    config.random_seeds = [123]
    config.num_iterations = 1  # todo
    config.batch_size = 1  # todo
    config.num_steps = 1  # number of steps for training transformation learning net
    config.deformator_type = 'ortho'  # choices = ['linear','ortho']
    config.num_directions = 5
    config.latent_dim = 512
    config.transformation_learning_net_lr = 0.0001
    config.epsilon = 6
    config.shift_distribution = 'normal'
    config.min_shift = 0.5
    config.label_weight = 1
    config.shift_weight = 0.25
    config.deformator_lr = 0.0001
    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config.train = True

    config.shifts_r = 5
    config.shifts_count = 2
    config.num_samples_lt = 2

    config.eval_samples = 8  # Number of samples used for computing re-scoring matrix
    config.eval_eps = 10  # Magnitude of perturbation for re-scoring analysis
    config.eval_directions = 2
    config.eval_batchsize = 2
    config.resume_direction = None

    config.saving_freq = 20
    config.logging_freq = 20
    config.evaluation_freq = 20
    config.visualisation_freq = 20

    config.classifier_path = 'pretrained_models'
    config.simple_cls_path = 'pretrained_models/classifiers/'
    config.nvidia_cls_path = 'pretrained_models/classifiers/nvidia_classifiers'
    config.image_path = 'data/celeba_hq/data1024x1024/'
    config.file_name = '/8_model.pkl'
    config.result_path = os.path.join('results', wandb.run.name, 'qualitative_analysis')

    return config
