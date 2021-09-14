import wandb


def get_config():
    run = wandb.init(project='disentangling_minor_factors', entity='kadarsh22')
    wandb.run.name = 'test_run'
    wandb.run.save()

    config = wandb.config
    config.gan_type = 'pggan'  # choices=['ProgGAN', 'StyleGAN2','SNGAN','StyleGAN']
    config.dataset = 'CelebAHQ'  # choices=['AnimeFaceS',CelebAHQ' ,'LSUN-cars', 'LSUN-cats'
    config.model_name = 'pggan_celebahq1024'  # choices = ['pggan_celebahq1024',stylegan_animeface512,stylegan_car512,stylegan_cat256]
    config.random_seed = 123
    config.iterations = 40001
    config.batch_size = 8
    config.deformator_type = 'ortho'  # choices = ['linear','ortho']
    config.deformator_lr = 0.0001
    config.num_directions = 512
    config.latent_dim = 512
    config.epsilon = 10
    config.device_name = 'cuda:'
    config.device_id = 0

    config.saving_freq = 2000
    config.logging_freq = 2000

    return config

