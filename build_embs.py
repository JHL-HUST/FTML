import numpy as np
import pickle
import os
import data_utils
import argparse


def main(argv=None):

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="./data/",
                        type=str,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default="imdb",
                        type=str,
                        # required=True,
                        help="The name of the task to train.")
    parser.add_argument("--max_candidates",
                        default=8,
                        type=int)
    parser.add_argument("--threshold_distance",
                        default=0.5,
                        type=float)
    parser.add_argument("--vocab_size",
                        default=50000,
                        type=int)
    parser.add_argument('--vGPU', type=str, default=None, help="Specify which GPUs to use.")

    args = parser.parse_args()

    if args.vGPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.vGPU

    MAX_VOCAB_SIZE = args.vocab_size
    if not os.path.exists(os.path.join(args.data_dir, 'aux_files')):
        os.makedirs(os.path.join(args.data_dir, 'aux_files'))

    # Generate dictionary
    if not os.path.isfile(os.path.join(args.data_dir, 'aux_files', 'org_dic_%s_%d.pkl' % (args.task_name, MAX_VOCAB_SIZE))):
        print('org_dic & org_inv_dic not exist, build and save the dict...')
        org_dic, org_inv_dic, _ = data_utils.build_dict(args.task_name, MAX_VOCAB_SIZE, data_dir=args.data_dir)
        if len(org_dic) < MAX_VOCAB_SIZE:
            MAX_VOCAB_SIZE = len(org_dic) - 1
        with open(os.path.join(args.data_dir, 'aux_files', 'org_dic_%s_%d.pkl' % (args.task_name, MAX_VOCAB_SIZE)), 'wb') as f:
            pickle.dump(org_dic, f, protocol=4)
        with open(os.path.join(args.data_dir, 'aux_files', 'org_inv_dic_%s_%d.pkl' % (args.task_name, MAX_VOCAB_SIZE)), 'wb') as f:
            pickle.dump(org_inv_dic, f, protocol=4)
    else:
        print('org_dic & org_inv_dic already exist, load the dict...')
        with open(os.path.join(args.data_dir, 'aux_files', 'org_dic_%s_%d.pkl' % (args.task_name, MAX_VOCAB_SIZE)), 'rb') as f:
            org_dic = pickle.load(f)
        with open(os.path.join(args.data_dir, 'aux_files', 'org_inv_dic_%s_%d.pkl' % (args.task_name, MAX_VOCAB_SIZE)), 'rb') as f:
            org_inv_dic = pickle.load(f)

    # Calculate the distance matrix
    if not os.path.isfile(os.path.join(args.data_dir, 'aux_files', 'small_dist_counter_%s_%d.npy' % (args.task_name, MAX_VOCAB_SIZE))):
        print('small dist counter not exists, create and save...')
        dist_mat = data_utils.compute_dist_matrix(org_dic, args.task_name, MAX_VOCAB_SIZE, data_dir=args.data_dir)
        print('dist matrix created!')
        small_dist_mat = data_utils.create_small_embedding_matrix(dist_mat, MAX_VOCAB_SIZE, threshold=1.5, retain_num=50)
        print('small dist counter created!')
        np.save(os.path.join(args.data_dir, 'aux_files', 'small_dist_counter_%s_%d.npy' % (args.task_name, MAX_VOCAB_SIZE)), small_dist_mat)
    else:
        print('small dist counter exists, loading...')
        small_dist_mat = np.load(os.path.join(args.data_dir, 'aux_files', 'small_dist_counter_%s_%d.npy' % (args.task_name, MAX_VOCAB_SIZE)))

    if not os.path.isfile(os.path.join(args.data_dir, 'aux_files', 'embeddings_glove_%s_%d.npy' % (args.task_name, MAX_VOCAB_SIZE))):
        print('embeddings glove not exists, creating...')
        glove_model = data_utils.loadGloveModel('glove.840B.300d.txt', data_dir=args.data_dir)
        glove_embeddings, _ = data_utils.create_embeddings_matrix(glove_model, org_dic, dataset=args.task_name, data_dir=args.data_dir)
        print("embeddings glove created!")
        np.save(os.path.join(args.data_dir, 'aux_files', 'embeddings_glove_%s_%d.npy' % (args.task_name, MAX_VOCAB_SIZE)), glove_embeddings)
    else:
        print('embeddings glove exists, loading...')
        glove_embeddings = np.load(os.path.join(args.data_dir, 'aux_files', 'embeddings_glove_%s_%d.npy' % (args.task_name, MAX_VOCAB_SIZE)))

if __name__ == '__main__':
    main()

