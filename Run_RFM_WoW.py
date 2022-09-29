from RFMWoWDataset import *
from RFM import *
from torch import optim
from trainers.DefaultTrainer import *
import torch.backends.cudnn as cudnn
import argparse
import os
from data.Utils_WoW import *


def train(args):
    data_path = 'dataset/wizard_of_wikipedia/'

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend='NCCL', init_method='env://')

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    batch_size = 32

    output_path = 'model/' + 'wizard_of_wikipedia/'

    vocab2id, id2vocab, id2freq = load_vocab(data_path + 'wow_input_output.vocab', t=args.min_vocab_freq)

    if not os.path.exists(data_path + 'glove.6B.300d.txt' + '.dat'):
        prepare_embeddings(data_path + 'glove.6B.300d.txt')
    emb_matrix = load_embeddings(data_path + 'glove.6B.300d.txt', id2vocab, args.embedding_size)

    samples, query, passage = load_default(args.dataset, data_path + args.dataset + '.answer',
                                           data_path + args.dataset + '.passage',
                                           data_path + args.dataset + '.pool',
                                           data_path + args.dataset + '.qrel',
                                           data_path + args.dataset + '.query')

    train_samples, dev_samples, test_seen_samples, test_unseen_samples = split_data(args.dataset,
                                                                                    data_path + args.dataset + '.split',
                                                                                    samples)
    print("The number of train_samples:", len(train_samples))

    train_dataset = RFMWoWDataset(vocab2id, args.mode, train_samples, query, passage, args.min_window_size,
                                  args.num_windows, args.knowledge_len, args.context_len)

    model = RFM(args.min_window_size, args.num_windows, args.embedding_size, args.knowledge_len, args.context_len,
                args.hidden_size, vocab2id, id2vocab, max_dec_len=70,
                beam_width=1, emb_matrix=emb_matrix)
    init_params(model, escape='embedding')

    model_optimizer = optim.Adam(model.parameters(), lr=0.0001)

    trainer = DefaultTrainer(model, args.local_rank)

    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue  # total parameters
        if param.requires_grad:
            Trainable_params += mulValue  # trainable parameters
        else:
            NonTrainable_params += mulValue  # non-trainable parameters

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

    # for i in range(10):
    #     trainer.train_epoch('ds_train', train_dataset, collate_fn, batch_size, i, model_optimizer)

    for i in range(30):
        if i == 0:
            train_embedding(model)
        trainer.train_epoch('fb_mle_mcc_ds_train', train_dataset, collate_fn, batch_size, i, model_optimizer)
        # multi_schedule.step()
        trainer.serialize(i, output_path=output_path)


def test(args):
    data_path = 'dataset/wizard_of_wikipedia/'

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    batch_size = 32

    output_path = 'model/' + 'wizard_of_wikipedia/'

    vocab2id, id2vocab, id2freq = load_vocab(data_path + 'wow_input_output.vocab', t=args.min_vocab_freq)

    samples, query, passage = load_default(args.dataset, data_path + args.dataset + '.answer',
                                           data_path + args.dataset + '.passage',
                                           data_path + args.dataset + '.pool',
                                           data_path + args.dataset + '.qrel',
                                           data_path + args.dataset + '.query')

    train_samples, dev_samples, test_seen_samples, test_unseen_samples = split_data(args.dataset,
                                                                                    data_path + args.dataset + '.split',
                                                                                    samples)
    print("The number of test_seen_samples:", len(test_seen_samples))
    print("The number of test_unseen_samples:", len(test_unseen_samples))

    test_seen_dataset = RFMWoWDataset(vocab2id, args.mode, test_seen_samples, query, passage, args.min_window_size,
                                      args.num_windows, args.knowledge_len, args.context_len)

    test_unseen_dataset = RFMWoWDataset(vocab2id, args.mode, test_unseen_samples, query, passage, args.min_window_size,
                                        args.num_windows, args.knowledge_len, args.context_len)

    for i in range(30):
        print('epoch ' + str(i))
        file = output_path + 'model/' + str(i) + '.pkl'

        if os.path.exists(file):
            model = RFM(args.min_window_size, args.num_windows, args.embedding_size, args.knowledge_len,
                        args.context_len, args.hidden_size, vocab2id, id2vocab, max_dec_len=70, beam_width=1)
            model.load_state_dict(torch.load(file))
            trainer = DefaultTrainer(model, None)
            # trainer.test('test', dev_dataset, collate_fn, batch_size, i, output_path=output_path)
            # seen
            print('test_seen:')
            trainer.test('test', test_seen_dataset, collate_fn, batch_size, 100 + i, output_path=output_path,
                         test_type=args.test)
            # unseen
            print('test_unseen:')
            trainer.test('test', test_unseen_dataset, collate_fn, batch_size, 1000 + i, output_path=output_path,
                         test_type=args.test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--test", type=str, default='SR')
    parser.add_argument("--dataset", type=str, default='wizard_of_wikipedia')
    parser.add_argument("--version", type=str, default='oracle')  # background version
    parser.add_argument("--embedding_size", type=int, default=300)  # embedding size
    parser.add_argument("--hidden_size", type=int, default=256)  # hidden size
    parser.add_argument("--min_window_size", type=int, default=4)  # the minimum size of slide window
    parser.add_argument("--num_windows", type=int, default=1)  # the stride of slide window
    parser.add_argument("--knowledge_len", type=int, default=256)  # background knowledge length
    parser.add_argument("--context_len", type=int, default=65)  # context length
    parser.add_argument("--min_vocab_freq", type=int, default=10)  # the minimum size of word frequency
    args = parser.parse_args()

    if args.mode == 'test':
        test(args)
    elif args.mode == 'train':
        train(args)
