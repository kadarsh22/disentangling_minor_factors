from model_loader import get_model
from train import Trainer
from saver import Saver
from utils import *
import wandb
from evaluation import Evaluator
from visualisation import Visualiser


def run_training_wrapper(config, logger, perf_logger):
    directories = list_dir_recursively_with_ignore('.',
                                                   ignores=['checkpoint.pt', '__pycache__', 'wandb', 'venv', '.idea',
                                                            'data', 'pretrained_models', 'data',
                                                            'results', '.git', '.gitignore', 'artifacts',
                                                            'pretrained_models.tar.xz', 'pretrained_models'])
    filtered_dirs = []
    for file in directories:
        x, y = file
        filtered_dirs.append((x, y))
    files = [(f[0], os.path.join(os.getcwd(), 'results', wandb.run.name, "src_copy", f[1])) for f in filtered_dirs]

    copy_files_and_create_dirs(files)

    models = get_model(config)
    model_trainer = Trainer(config)
    saver = Saver(config)
    # evaluator = Evaluator(config)
    visualiser = Visualiser(config)

    source_generator, source_deformator, target_generator, target_deformator, target_deformator_opt, transformation_learning_net, \
    transformation_learning_net_opt = models
    source_generator.eval()
    target_generator.eval()
    source_deformator.eval()
    target_deformator.to(config.device).train()
    # saver.load_model((deformator,deformator_opt))
    transformation_learning_net = model_trainer.train_transformation_learning_net(source_generator, source_deformator,
                                                                                  transformation_learning_net,
                                                                                  transformation_learning_net_opt)

    logit_loss_list = []
    shift_loss_list = []
    visualiser.generate_latent_traversal_pggan(source_generator, source_deformator, 'reference')

    for iteration in range(config.num_deformator_iterations):
        if iteration % config.saving_freq == 0:
            params = (target_deformator, target_deformator_opt, transformation_learning_net)
            perf_logger.start_monitoring("Saving Model for iteration :" + str(iteration))
            saver.save_model(params, iteration)
            perf_logger.stop_monitoring("Saving Model for iteration :" + str(iteration))

        target_deformator, target_deformator_opt, logit_loss, shift_loss = \
            model_trainer.train_ours(target_generator, target_deformator, target_deformator_opt,
                                     transformation_learning_net)
        logit_loss_list.append(logit_loss)
        shift_loss_list.append(shift_loss)

        if iteration % config.logging_freq == 0 and iteration != 0:
            logit_loss_avg = sum(logit_loss_list) / len(logit_loss_list)
            shift_loss_avg = sum(shift_loss_list) / len(shift_loss_list)
            logger.info("step : %d / %d deformator_logit_loss : %.4f deformator_shift_loss  %.4f " % (
                iteration, config.num_deformator_iterations, logit_loss_avg, shift_loss_avg))
            wandb.log({'num_deformator_iterations': iteration + 1, 'deformator_logit_loss': logit_loss_avg,
                       'deformator_shift_loss': shift_loss_avg})
            logit_loss_list = []
            shift_loss_list = []

        if iteration % config.visualisation_freq == 0:
            perf_logger.start_monitoring("Visualising model for iteration :" + str(iteration))
            visualiser.generate_latent_traversal_stylegan(target_generator, target_deformator, iteration)
            perf_logger.stop_monitoring("Visualising model for iteration :" + str(iteration))
