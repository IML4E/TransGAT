import torch
import torch.nn as nn
import numpy as np
import torch.utils.data.distributed

from models import SpKBGATModified, SpKBGATConvOnly

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus

import argparse
import os
import pickle

def parse_args_kinship():
    args = argparse.ArgumentParser()

    args.add_argument("-dataset_name", "--dataset_name",
                      default="kinship", help="name of dataset")

    # network arguments
    args.add_argument("-patience_gat", "--patience_gat",
                      default=1200, help="early stopping patience of GAT")
    args.add_argument("-patience_conv", "--patience_conv",
                      default=200, help="early stopping patience of ConvKB")
    args.add_argument("-patience", "--patience",
                      default=25, help="early stopping patience")
    args.add_argument("-data", "--data",
                      default="../data/kinship/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=400, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=1e-5, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=400, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/kinship/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=8544, help="Batch size for GAT")  # 86835
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[200, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=1, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=10,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=50,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.1, help="Dropout probability for convolution layer")

    args = args.parse_args()
    return args

def parse_args_fb():
    args = argparse.ArgumentParser()

    args.add_argument("-dataset_name", "--dataset_name",
                      default="fb", help="name of dataset")

    # network arguments
    args.add_argument("-patience", "--patience",
                      default=25, help="early stopping patience")
    args.add_argument("-data", "--data",
                      default="../data/FB15k-237/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=1e-5, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=200, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-4)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/FB15k-237/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=103, help="Batch size for GAT")  # 86835
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=1, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=50,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.3, help="Dropout probability for convolution layer")
    args = args.parse_args()
    return args

def parse_args_WN18_transR():
    args = argparse.ArgumentParser()

    args.add_argument("-dataset_name", "--dataset_name",
                      default="WN18_transR", help="name of dataset")

    # network arguments
    args.add_argument("-patience", "--patience",
                      default=50, help="early stopping patience")
    args.add_argument("-data", "--data",
                      default="../data/WN18_transR/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=200, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-4)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/WN18_transR/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=2003, help="Batch size for GAT")  # 86835
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.003, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=1, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layer")

    args = args.parse_args()
    return args

def parse_args_WN18RR():
    args = argparse.ArgumentParser()

    args.add_argument("-dataset_name", "--dataset_name",
                      default="WN18RR", help="name of dataset")

    # network arguments
    args.add_argument("-patience_gat", "--patience_gat",
                      default=120, help="early stopping patience of GAT")
    args.add_argument("-patience_conv", "--patience_conv",
                      default=50, help="early stopping patience of ConvKB")
    args.add_argument("-data", "--data",
                      default="../data/WN18RR/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3600, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=250, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=200, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/WN18RR/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=86835, help="Batch size for GAT")  # 86835
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.1, help="Dropout probability for convolution layer")

    args = args.parse_args()
    return args

def parse_args_umls():
    args = argparse.ArgumentParser()

    args.add_argument("-dataset_name", "--dataset_name",
                      default="umls", help="name of dataset")

    # network arguments
    args.add_argument("-patience", "--patience",
                      default=2400, help="early stopping patience")
    args.add_argument("-data", "--data",
                      default="../data/umls/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=400, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=1e-5, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=200, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/umls/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=5316, help="Batch size for GAT")  # 86835
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=3, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=10,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=50,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.3, help="Dropout probability for convolution layer")

    args = args.parse_args()
    return args

def load_data(args):
    train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
        args.data, is_unweigted=False, directed=True)
    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                                 os.path.join(args.data, 'relation2vec.txt'))
        print("Initialised relations and entities from TransE")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")
    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)
    if (args.get_2hop):
        file = args.data + "/2hop.pickle"
        with open(file, 'wb') as handle:
            pickle.dump(corpus.node_neighbors_2hop, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    if (args.use_2hop):
        print("Opening node_neighbors pickle object")
        file = args.data + "/2hop.pickle"
        with open(file, 'rb') as handle:
            node_neighbors_2hop = pickle.load(handle)
    # return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)
    return corpus, torch.cuda.FloatTensor(entity_embeddings), torch.cuda.FloatTensor(
        relation_embeddings), node_neighbors_2hop

args_list = [parse_args_umls()] #, parse_args_WN18RR(), parse_args_WN18_transR(), parse_args_fb()
torch.cuda.set_device(3)

for args in args_list:
    Corpus_, entity_embeddings, relation_embeddings, node_neighbors_2hop = load_data(args)

    '''
    增加的nhop_embedding索引
    '''
    nhop_set = set()
    nhop_indices = Corpus_.get_batch_nhop_neighbors_all(args, Corpus_.unique_entities_train, node_neighbors_2hop)
    for edge in nhop_indices:
        nhop_set.add(str(edge[1]) + "_" + str(edge[2]))
    nhop_embedding_dic = {}
    nhop_array = []
    nhop_num = 0
    for nhop_edge in nhop_set:
        nhop_embedding_dic[nhop_edge] = nhop_num
        nhop_array.append(nhop_edge.split('_'))
        nhop_num += 1
    nhop_embeddings = torch.cuda.FloatTensor(np.random.randn(nhop_num, args.embedding_size))
    nhop_embeddings = nn.Parameter(nhop_embeddings)
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                    args.drop_GAT, args.alpha, args.nheads_GAT)
    model_gat.load_state_dict(torch.load('{}trained.pth'.format(args.output_folder)))

    for i in range(len(nhop_array)):
        nhop_embeddings[i] = model_gat.final_relation_embeddings[int(nhop_array[i][0])] + model_gat.final_relation_embeddings[int(nhop_array[i][1])]

    dir = {'array': nhop_array, 'embeddings':nhop_embeddings, 'relation_embeddings': model_gat.final_relation_embeddings}

    file_name = args.dataset_name + "_2hop.pickle"
    file_path = args.data + "/" + file_name
    print(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(dir, f)
    with open(file_path, 'rb') as f:
        dir_red = pickle.load(f)
