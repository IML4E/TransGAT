import torch
import torch.nn as nn
import torch.utils.data.distributed
from ignite.metrics import Loss

from models import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from copy import deepcopy

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus

import random
import argparse
import os
import time
import pickle
from tensorboardX import SummaryWriter

# %%
# %%from torchviz import make_dot, make_dot_from_trace

writer = SummaryWriter("./kinship_testing/")

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-patience_gat", "--patience_gat",
                      default=120, help="early stopping patience of GAT")
    args.add_argument("-patience_conv", "--patience_conv",
                      default=60, help="early stopping patience of ConvKB")
    args.add_argument("-data", "--data",
                      default="./kinship/", help="data directory")
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
                      default=0.3, help="Dropout probability for convolution layer")

    args = args.parse_args()
    return args


args = parse_args()
# %%
print("start load data")
torch.cuda.set_device(1)


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

    '''
    entity_embeddings = np.random.randn(
        len(entity2id), args.embedding_size)
    relation_embeddings = np.random.randn(
        len(relation2id), args.embedding_size)
    print("Initialised relations and entities randomly")
    '''

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


Corpus_, entity_embeddings, relation_embeddings, node_neighbors_2hop = load_data(args)


entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)

print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size()))
# %%

CUDA = torch.cuda.is_available()


def save_model(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained.pth"))
    print("Done saving Model")


def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def get_unique_entity(indices):
    unique_entities = set()
    for source in indices:
        e1 = source[0]
        e2 = source[2]
        unique_entities.add(e1)
        unique_entities.add(e2)
    return list(unique_entities)
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
def train_gat(args):
    # Creating the gat model here.
    ####################################
    print("Defining model")
    print("\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))


    '''
    model_gat初始化进行修改
    '''
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, nhop_embeddings, nhop_array, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)
    if CUDA:
        model_gat = model_gat.cuda()


    optimizer = torch.optim.Adam(model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    # train_loader = torch.utils.data.DataLoader(
    #     current_batch_2hop_indices,  sampler=train_sampler, **kwargs)
    epoch_losses = []  # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))

    """
        Add by cc
        Add early stopping with patience
        if patience times in a row score_fn gives a result lower than the best result,
        than the training will be stopped
        """
    counter = 0
    best_score = None
    for epoch in range(args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []


        if len(Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (len(Corpus_.train_indices) // args.batch_size_gat) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)
            iter_unique_entities = get_unique_entity(train_indices)
            current_batch_2hop_indices = torch.tensor([])
            start_2hop_time = time.time()
            current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args, iter_unique_entities,
                                                                              node_neighbors_2hop)
            end_2hop_time = time.time()
            print(end_2hop_time - start_2hop_time)
            if CUDA:
                train_indices = Variable(torch.LongTensor(train_indices)).cuda()
                # train_values = Variable(torch.FloatTensor(train_values)).cuda().half()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                # train_values = Variable(torch.FloatTensor(train_values))

            current_batch_2hop_indices = Variable(torch.LongTensor(current_batch_2hop_indices)).cuda()

            # forward pass
            entity_embed, relation_embed = model_gat(
                Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)

            optimizer.zero_grad()

            loss = batch_gat_loss(
                gat_loss_func, train_indices, entity_embed, relation_embed)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, avg_loss, time.time() - start_time))
        epoch_losses.append(avg_loss)

        """
        early stopping
        """
        if best_score is None:
            best_score = 99
        elif avg_loss > best_score:
            counter += 1
            if counter >= args.patience_gat:
                break
        else:
            best_score = avg_loss
            counter = 0

        # plot avg_loss
        writer.add_scalar('WN18RR_testing [add (h,t)->r]: GAT average loss--epoch', sum(epoch_loss) / len(epoch_loss), epoch)
        save_model(model_gat, args.data, epoch, args.output_folder)


def train_conv(args):
    # Creating convolution model here.
    ####################################
    print("Defining model")
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, nhop_embeddings, nhop_array, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)
    print("Only Conv model trained")
    '''
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
                                 '''
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, nhop_embeddings, args.entity_out_dim,
                                 args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
    if CUDA:
        model_conv.cuda()
        model_gat.cuda()

    model_gat.load_state_dict(torch.load(
        '{}trained.pth'.format(args.output_folder)))
    model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    model_conv.final_relation_embeddings = model_gat.final_relation_embeddings_new

    Corpus_.batch_size = args.batch_size_conv
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

    optimizer = torch.optim.Adam(model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()

    epoch_losses = []  # losses of all epochs
    print("Number of epochs {}".format(args.epochs_conv))

    """
            Add by cc
            Add early stopping with patience
            if patience times in a row score_fn gives a result lower than the best result,
            than the training will be stopped
            """
    counter = 0
    best_score = None
    for epoch in range(args.epochs_conv):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(Corpus_.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (len(Corpus_.train_indices) // args.batch_size_conv) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(torch.LongTensor(train_indices)).cuda(1)
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            preds = model_conv(
                Corpus_, Corpus_.train_adj_matrix, train_indices)

            optimizer.zero_grad()

            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))
            writer.add_scalar('WN18RR_add_conv_epoch [add (h,t)->r]: ConvKB per iteration loss--iter', loss.data.item(), iters)
        scheduler.step()
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, avg_loss, time.time() - start_time))
        epoch_losses.append(avg_loss)

        """
                early stopping
                """
        if best_score is None:
            best_score = 99
        elif avg_loss > best_score:
            counter += 1
            if counter >= args.patience_conv:
                break
        else:
            best_score = avg_loss
            counter = 0

        writer.add_scalar('WN18RR_add_conv_epoch [add (h,t)->r]: ConvKB average loss--epoch', sum(epoch_loss) / len(epoch_loss), epoch)
        save_model(model_conv, args.data, epoch, args.output_folder + "conv/")


def evaluate_conv(args, unique_entities):
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, nhop_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
    model_conv.load_state_dict(torch.load(
        '{}conv/trained.pth'.format(args.output_folder)))

    model_conv.cuda()
    model_conv.eval()
    with torch.no_grad():
        Corpus_.get_validation_pred(args, model_conv, unique_entities)

train_gat(args)
train_conv(args)
evaluate_conv(args, Corpus_.unique_entities_train)

# 导出并关闭TensorboardX
writer.export_scalars_to_json("./tanh_Hadam.json")
writer.close()
