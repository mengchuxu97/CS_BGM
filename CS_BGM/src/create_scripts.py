"""Create scripts for experiments"""

import os
import re
import copy
import itertools
from argparse import ArgumentParser
import utils

BASE_SCRIPT = [
    "python -u ./src/compressed_sensing.py \\",
    "    --pretrained-model-dir ./models/celebA_64_64/ \\",
    "    \\",
    "    --dataset celebA \\",
    "    --input-type full-input \\",
    "    --num-input-images 64 \\",
    "    --batch-size 64 \\",
    "    \\",
    "    --measurement-type project \\",
    "    --noise-std 0.01 \\",
    "    --num-measurements 50 \\",
    "    \\",
    "    --model-types k-sparse-wavelet \\",
    "    --mloss1_weight 0.0 \\",
    "    --mloss2_weight 1.0 \\",
    "    --zprior_weight 0.001 \\",
    "    --dloss1_weight 0.0 \\",
    "    --dloss2_weight 0.0 \\",
    "    --theta_loss_weight 0.0 \\",
    "    --lmbd 0.1 \\",
    "    --sparsity 50 \\",
    "    \\",
    "    --optimizer-type adam \\",
    "    --learning-rate 0.1 \\",
    "    --momentum 0.9 \\",
    "    --max-update-iter 500 \\",
    "    --max-update-iter-theta 400\\",
    "    --num-random-restarts 2 \\",
    "    --sample_num_z 10 \\",
    "    \\",
    "    --save-images \\",
    "    --save-stats \\",
    "    --print-stats \\",
    "    --checkpoint-iter 1 \\",
    "    --image-matrix 0",
]

# Return the idx of the first str in list_of_strs that
# contains str_to_search as a substring (returns None in
# case of no match)
def find_overlap_idx(list_of_strs, str_to_search):
    for (idx, cur_str) in enumerate(list_of_strs):
        if str_to_search in cur_str:
            return idx


def get_setting_specific_name(setting_dict):
    keys_to_ignore = ['script_type', 'task', 'sequence_length', 'cell_type']
    keys_in_filename = list(set(setting_dict.keys()).difference(keys_to_ignore))
    keys_in_filename.sort()
    filename = '_'.join([str(key) + '_' + str(setting_dict[key]) for key in keys_in_filename])
    return filename


def create_script(setting_dict, hparams):
    # Copy base script text
    script_text = copy.deepcopy(BASE_SCRIPT)

    # Change other hyperparams
    other_hparam_names = set(setting_dict.keys())
    for hparam_name in other_hparam_names:
        idx = find_overlap_idx(script_text, hparam_name)
        script_text[idx] = '    --' + hparam_name + '=' + setting_dict[hparam_name] + ' \\'
    idx = find_overlap_idx(script_text, 'dataset')
    script_text[idx] = '    --' + 'dataset' + '=' + hparams.dataset + ' \\'
    idx = find_overlap_idx(script_text, 'num-input-images')
    script_text[idx] = '    --' + 'num-input-images' + '=' + str(hparams.num_input_images) + ' \\'
    idx = find_overlap_idx(script_text, 'pretrained-model-dir')
    script_text[idx] = '    --' + 'pretrained-model-dir' + '=' + hparams.pretrained_model_dir + ' \\'
    idx = find_overlap_idx(script_text, 'batch-size')
    script_text[idx] = '    --' + 'batch-size' + '=' + str(hparams.batch_size) + ' \\'
    idx = find_overlap_idx(script_text, 'sample_num_z')
    script_text[idx] = '    --' + 'sample_num_z' + '=' + str(hparams.sample_num_z) + ' \\'
    idx = find_overlap_idx(script_text, 'max-update-iter-theta')
    script_text[idx] = '    --' + 'max-update-iter-theta' + '=' + str(hparams.max_update_iter_theta) + ' \\'


    end_file_name = get_setting_specific_name(setting_dict) + '.sh'
    script_file_name = reduce(os.path.join, [hparams.scripts_base_dir, end_file_name])

    # Remove trailing / if present
    if script_text[-1].endswith('\\'):
        script_text[-1] = script_text[-1][:-1]

    writer = open(script_file_name, 'w+')
    writer.write('\n'.join(script_text))
    writer.close()


def create_scripts(hparams, hparam_names_for_grid_search):
    keys = hparam_names_for_grid_search

    hparam_lists_to_combine = []
    for hparam_name in hparam_names_for_grid_search:
        hparams_attr_name = re.sub('-', '_', hparam_name)
        hparam_lists_to_combine.append(getattr(hparams, hparams_attr_name))

    for setting in itertools.product(*hparam_lists_to_combine):
        setting_dict = dict()
        for (idx, value) in enumerate(setting):
            setting_dict[keys[idx]] = value
        create_script(setting_dict, hparams)


if __name__ == '__main__':
    PARSER = ArgumentParser()

    # Hyperparameters for models (command line options from main.py)
    # Multi word hyperparameter names should be hyphen separated
    HPARAM_NAMES_FOR_GRID_SEARCH = ['input-type',
                                    'measurement-type',
                                    'noise-std',
                                    'num-measurements',
                                    'model-types',
                                    'zprior_weight',
                                    'dloss1_weight',
                                    'theta_loss_weight',
                                    'lmbd',
                                    'max-update-iter',
                                    'num-random-restarts',
                                    'sparsity',
                                   ]

    for hparam in HPARAM_NAMES_FOR_GRID_SEARCH:
        PARSER.add_argument('--' + hparam, metavar=hparam+'-val', type=str, nargs='+',
                            default=['0'], help='Values of ' + hparam)

    PARSER.add_argument('--scripts-base-dir', type=str, default='../scripts/',
                        help='Base directory to save scripts: Absolute path or relative to src')
    PARSER.add_argument('--batch-size', type=int, default=64, help='Batch size')
    PARSER.add_argument('--dataset', type=str, default='celebA')
    PARSER.add_argument('--num-input-images', type=int, default=64)
    PARSER.add_argument('--pretrained-model-dir', type=str, default='./models/celebA_64_64/')
    PARSER.add_argument('--sample_num_z', type=int, default=10)
    PARSER.add_argument('--max-update-iter-theta', type=int, default=400)
    HPARAMS = PARSER.parse_args()
    utils.print_hparams(HPARAMS)

    create_scripts(HPARAMS, HPARAM_NAMES_FOR_GRID_SEARCH)
