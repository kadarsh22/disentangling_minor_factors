from torch.utils.data import DataLoader
import numpy as np
import random
import torch
import os
from models.closedform.utils import load_generator
from utils import NoiseDataset
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn.functional as F
from logger import PerfomanceLogger
import seaborn as sns
from models.attribute_predictors import attribute_predictor, attribute_utils

sns.set_theme()
perf_logger = PerfomanceLogger()

class Evaluator(object):
    def __init__(self, config):
        self.config = config
        self.result_path = config.result_path
        self.simple_cls_path = config.simple_cls_path
        self.nvidia_cls_path = config.nvidia_cls_path
        self.directions_idx = list(range(2))  # [4, 16, 23, 24, 8, 11]  ##TODOD change from 0 to 512
        self.num_directions = len(self.directions_idx)
        self.num_samples = config.eval_samples
        self.epsilon = config.eval_eps
        self.z_batch_size = config.batch_size
        self.num_batches = int(self.num_samples / self.z_batch_size)
        self.all_attr_list = ['eyeglasses', 'Bald', 'Bangs', 'Goatee', 'Mustache', 'Blurry', 'Pale_Skin',
                              'Wearing_Lipstick']
        attr_index = list(range(len(self.all_attr_list)))
        self.attr_list_dict = OrderedDict(zip(self.all_attr_list, attr_index))

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _get_predictor_list(self, attr_list):
        predictor_list = []
        for each in attr_list[:1]:
            predictor = attribute_predictor.get_classifier(
                os.path.join(self.simple_cls_path, each, "weight.pkl"),self.config.device)
            predictor.to(self.config.device).eval()
            predictor_list.append(predictor)
        for classifier_name in attr_list[1:]:
            predictor = attribute_utils.ClassifierWrapper(classifier_name, ckpt_path=self.nvidia_cls_path,
                                                          device=self.config.device)
            predictor.to(self.config.device).eval()
            predictor_list.append(predictor)
        return predictor_list

    def get_reference_attribute_scores(self, generator, z_loader, attribute_list):
        predictor_list = self._get_predictor_list(attribute_list)
        ref_image_scores = []
        with torch.no_grad():
            for batch_idx, z in enumerate(z_loader):
                images = generator(z)
                images = (images + 1) / 2
                predict_images = F.avg_pool2d(images, 4, 4)
                for predictor_idx, predictor in enumerate(predictor_list):
                    ref_image_scores.append(torch.softmax(
                        predictor(predict_images), dim=1)[:, 1])
        ref_image_scores = torch.stack(ref_image_scores).view(len(predictor_list), -1)
        ref_image_scores = torch.stack([ref_image_scores.view(-1, self.z_batch_size)[i::len(predictor_list)] for i in
                                        range(len(predictor_list))]).view(len(predictor_list), self.num_samples)
        ref_image_scores = ref_image_scores.unsqueeze(0).repeat(len(self.directions_idx), 1, 1)
        torch.save(ref_image_scores, os.path.join(self.result_path, 'reference_attribute_scores.pkl'))
        return ref_image_scores

    def get_evaluation_metric_values(self, generator, deformator, attribute_list, reference_attr_scores, z_loader,
                                     directions_idx, resume=False, direction_to_resume=None):
        predictor_list = self._get_predictor_list(attribute_list)
        if not resume:
            shifted_image_scores = []
        else:
            shifted_image_scores = torch.load(os.path.join(self.result_path, 'shifted_scores_intermediate.pkl'))
            directions_idx = list(range(direction_to_resume, self.num_directions))
        with torch.no_grad():
            for dir_index, dir in enumerate(directions_idx):
                perf_logger.start_monitoring("Direction " + str(dir) + " completed")
                for batch_idx, z in enumerate(z_loader):
                    z_shift = z + deformator[dir: dir + 1] * self.epsilon
                    images_shifted = generator(z_shift)
                    images_shifted = (images_shifted + 1) / 2
                    predict_images = F.avg_pool2d(images_shifted, 4, 4)
                    for predictor_idx, predictor in enumerate(predictor_list):
                        shifted_image_scores.append(torch.softmax(
                            predictor(predict_images), dim=1)[:, 1])
                if dir % 25 == 0:
                    torch.save(shifted_image_scores, os.path.join(self.result_path, 'shifted_scores_intermediate.pkl'))
                perf_logger.stop_monitoring("Direction " + str(dir) + " completed")

        shifted_image_scores = torch.stack(shifted_image_scores).view(len(self.directions_idx), len(predictor_list), -1)
        shifted_image_scores = torch.stack(
            [shifted_image_scores.view(-1, self.z_batch_size)[i::len(predictor_list)] for i in
             range(len(predictor_list))]).view(len(predictor_list), self.num_directions, self.num_samples).permute(1, 0,
                                                                                                                   2)

        difference_matrix = shifted_image_scores - reference_attr_scores
        rescoring_matrix = np.round(torch.abs(torch.mean(difference_matrix, dim=-1)).cpu().numpy(), 2)
        all_predictions = (shifted_image_scores > 0.5).float()
        all_dir_attr_manipulation_acc = all_predictions.mean(dim=-1).cpu().numpy()

        torch.save(rescoring_matrix, os.path.join(self.result_path, 'rescoring matrix.pkl'))
        torch.save(all_dir_attr_manipulation_acc,
                   os.path.join(self.result_path, 'attribute manipulation accuracy.pkl'))
        # self.get_heat_map(rescoring_matrix, directions_idx, attribute_list, self.result_path)
        return rescoring_matrix, all_dir_attr_manipulation_acc

    @staticmethod
    def get_heat_map(matrix, dir, attribute_list, path, labels, classifier='full'):
        sns.set(font_scale=1.8)
        sns.set(font='Times New Roman')
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.rcParams['font.family'] = "Times New Roman"
        hm = sns.heatmap(matrix, annot=True, fmt=".2f", cbar=False, cmap='Blues')
        ax.xaxis.tick_top()
        plt.xticks(np.arange(len(attribute_list)) + 0.5, labels=attribute_list)
        plt.yticks(np.arange(len(dir)) + 0.5, labels=labels , rotation=0)
        plt.tick_params(top=False)
        plt.tight_layout()
        plt.savefig(os.path.join(path, classifier + '_Rescoring_Analysis' + '.svg'), dpi=300)
        plt.close('all')

    @staticmethod
    def get_partial_metrics(result_path, attributes, direction_idx, attr_list_dict, attr_vs_direction, rescoring_matrix,
                            labels, attr_manipulation_acc):
        dir_attr_path = os.path.join(result_path, 'Direction_vs_Classifier_Metrics')
        os.makedirs(dir_attr_path, exist_ok=True)
        selected_attr = {cls_key: attr_list_dict[cls_key] for cls_key in attributes}
        attr_indices = list(selected_attr.values())
        temp_matrix = rescoring_matrix[direction_idx]
        partial_rescoring_matrix = temp_matrix[:, attr_indices]
        Evaluator.get_heat_map(partial_rescoring_matrix, direction_idx, attributes, dir_attr_path, labels,
                               classifier='partial')
        for cls, dir in attr_vs_direction.items():
            acc = attr_manipulation_acc[dir, attr_vs_direction[cls]]
            attr_vs_direction[cls] = eval(str(acc))

        with open(os.path.join(dir_attr_path, 'Attribute_manipulation_accuracies_partial.json'),
                  'w') as fp:
            json.dump(attr_vs_direction, fp)
        return partial_rescoring_matrix, attr_vs_direction

    def get_classifer_analysis(self, attributes, dir_idx, top_k, rescoring_matrix, attr_manipulation_acc):
        selected_attr = {cls_key: self.attr_list_dict[cls_key] for cls_key in attributes}
        for cls, cls_index in selected_attr.items():
            classifier_direction_dict = {}
            classifier_analysis_result_path = os.path.join(self.result_path, cls)
            os.makedirs(classifier_analysis_result_path, exist_ok=True)
            rescoring_matrix = torch.FloatTensor(rescoring_matrix)
            classifier_variance = rescoring_matrix[:, cls_index]
            best_direction_indices = torch.sort(classifier_variance, descending=True)[1][0:top_k]
            top_k_directions = np.array(dir_idx)[best_direction_indices]
            top_k_directions = getattr(top_k_directions, "tolist", lambda: top_k_directions)()
            top_k_direction_acc = attr_manipulation_acc[:, cls_index][best_direction_indices]
            top_k_direction_acc = getattr(top_k_direction_acc, "tolist", lambda: top_k_direction_acc)()
            classifier_rescoring_matrix = rescoring_matrix[best_direction_indices]
            classifier_direction_dict[cls] = {'top_directions': top_k_directions,
                                              'top_directions attr manipulation accuracy': top_k_direction_acc}

            torch.save(classifier_rescoring_matrix,
                       os.path.join(classifier_analysis_result_path, cls + '_rescoring_matrix.pkl'))
            labels = [str(direction) for direction in self.directions_idx]
            self.get_heat_map(classifier_rescoring_matrix, top_k_directions, attributes,
                              classifier_analysis_result_path, labels, classifier=cls)
            with open(os.path.join(classifier_analysis_result_path, 'Classifier_top_directions_details.json'),
                      'w') as fp:
                json.dump(classifier_direction_dict, fp)
            # print('Classifier analysis for ' + cls + ' at index ' + str(cls_index) + ' completed!!')

    def evaluate_directions(self, generator, deformator, resume=False, resume_dir=None):
        if not resume:
            codes = torch.randn(self.num_samples, generator.z_space_dim).to(self.config.device)
            codes = generator.layer0.pixel_norm(codes)
            codes = codes.detach()
            z = NoiseDataset(latent_codes=codes, num_samples=self.num_samples, z_dim=generator.z_space_dim)
            os.makedirs(self.result_path,exist_ok=True)
            torch.save(z, os.path.join(self.result_path, 'z_analysis.pkl'))
        else:
            z = torch.load(os.path.join(self.result_path, 'z_analysis.pkl'))
        z_loader = DataLoader(z, batch_size=self.z_batch_size, shuffle=False)
        perf_logger.start_monitoring("Reference attribute scores done")
        reference_attr_scores = self.get_reference_attribute_scores(generator, z_loader, self.all_attr_list)
        perf_logger.stop_monitoring("Reference attribute scores done")
        perf_logger.start_monitoring("Metrics done")
        full_rescoring_matrix, full_attr_manipulation_acc = self.get_evaluation_metric_values(generator, deformator,
                                                                                              self.all_attr_list,
                                                                                              reference_attr_scores,
                                                                                              z_loader,
                                                                                              self.directions_idx,
                                                                                              resume=resume,
                                                                                              direction_to_resume=resume_dir)
        perf_logger.stop_monitoring("Metrics done")
        classifiers_to_analyse = self.all_attr_list
        top_k = 5
        self.get_classifer_analysis(classifiers_to_analyse, self.directions_idx, top_k, full_rescoring_matrix,
                                    full_attr_manipulation_acc)
