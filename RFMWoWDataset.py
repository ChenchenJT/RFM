from torch.utils.data import Dataset
from data.Utils import *
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def get_selection_label(b, r, min_window_size=5, n_windows=4):
    # print(b.size())
    window_size = min_window_size
    bs = list()
    for i in range(n_windows):
        bs.append(F.pad(b.unfold(1, window_size, min_window_size), (0, min_window_size * n_windows - window_size)))
        window_size += min_window_size
    b_segments = torch.cat(bs, dim=1)

    b_list = b_segments.tolist()
    r_list = r.tolist()

    overlap = [[len(set(seg).intersection(r_list[i])) for seg in b_list[i]] for i in range(len(b_list))]

    p_s = F.softmax(torch.tensor(overlap).float(), dim=-1).detach()
    return p_s


class RFMWoWDataset(Dataset):
    def __init__(self, vocab2id, mode, samples, query, passage, min_window_size=5, num_windows=4, knowledge_len=300,
                 context_len=65, max_dec_length=80, n=1E10):
        super(RFMWoWDataset, self).__init__()

        self.min_window_size = min_window_size
        self.num_windows = num_windows
        self.knowledge_len = knowledge_len
        self.context_len = context_len
        self.max_dec_length = max_dec_length

        # WoW
        self.mode = mode
        self.samples = samples
        self.query = query
        self.passage = passage
        self.query_id = list()
        self.context_id = list()

        # 标量
        self.ids = list()
        self.contexts = list()
        self.queries = list()
        self.responses = list()
        self.unstructured_knowledges = list()
        self.dyn_vocab2ids = list()
        self.dyn_id2vocabs = list()
        self.example_id = list()

        # tensor
        self.id_arrays = list()
        self.context_arrays = list()
        self.query_arrays = list()
        self.response_arrays = list()
        self.dyn_response_arrays = list()
        self.unstructured_knowledge_arrays = list()

        self.ref_start_arrays = list()
        self.ref_end_arrays = list()

        self.dyn_map_arrays = list()
        self.vocab_map_arrays = list()
        self.vocab_overlap_arrays = list()

        self.selections = list()

        self.vocab2id = vocab2id
        self.n = n

        self.load()

    def load(self):
        for id in range(len(self.samples)):
            sample = self.samples[id]

            # 处理对话上下文
            context = self.query[sample['query_id']]
            context_ = [word.lower() for word in context]
            self.contexts.append(context_)
            if len(context_) > self.context_len:
                context_ = context_[-self.context_len:]
            elif len(context_) < self.context_len:
                context_ = context_ + [PAD_WORD] * (self.context_len - len(context_))
            self.context_arrays.append(torch.tensor(
                [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in context_],
                requires_grad=False).long())

            # 处理背景知识
            # correct answer is always the first one
            temp_sample_knowledge_pool = sample['shifting_knowledge_pool'].copy()
            unstructured_knowledge_origin = []
            for pid in temp_sample_knowledge_pool:
                temp = self.passage[pid]
                for word in temp:
                    unstructured_knowledge_origin.append(word)
                if len(unstructured_knowledge_origin) > 256:
                    break
            unstructured_knowledge_origin = [word.lower() for word in unstructured_knowledge_origin]
            self.unstructured_knowledges.append(unstructured_knowledge_origin)
            unstructured_knowledge = unstructured_knowledge_origin
            if len(unstructured_knowledge) > self.knowledge_len:
                unstructured_knowledge = unstructured_knowledge[:self.knowledge_len]
            else:
                unstructured_knowledge = unstructured_knowledge + [PAD_WORD] * (self.knowledge_len - len(unstructured_knowledge))
            b = torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in
                              unstructured_knowledge], requires_grad=False).long()
            if b.size()[0] > 256 or b.size()[0] == 0:
                print(len(unstructured_knowledge))
            self.unstructured_knowledge_arrays.append(b)

            # 处理ground-true背景知识
            bg_ref_start = -1
            bg_ref_end = -1
            if temp_sample_knowledge_pool[0] != 'K_0':
                bg_ref_start = 0
                bg_ref_end = len(self.passage[temp_sample_knowledge_pool[0]]) - 1
            self.ref_start_arrays.append(torch.tensor([bg_ref_start], requires_grad=False))
            self.ref_end_arrays.append(torch.tensor([bg_ref_end], requires_grad=False))

            # 处理回复
            response = sample['response']
            response = [word.lower() for word in response]
            self.responses.append([response])
            response = (response + [EOS_WORD])[:self.max_dec_length]
            r = torch.tensor(
                [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in response],
                requires_grad=False).long()
            self.response_arrays.append(r)

            self.selections.append(
                get_selection_label(b.unsqueeze(0), r.unsqueeze(0), min_window_size=self.min_window_size,
                                    n_windows=self.num_windows))

            # from data.Utils import build_vocab
            dyn_vocab2id, dyn_id2vocab = build_vocab(unstructured_knowledge)  # 返回知识的vocab2id和id2vocab，特殊字符只有PAD
            self.dyn_vocab2ids.append(dyn_vocab2id)
            self.dyn_id2vocabs.append(dyn_id2vocab)

            self.dyn_response_arrays.append(
                torch.tensor([dyn_vocab2id.get(w) if w in dyn_vocab2id else 0 for w in response],
                             requires_grad=False).long())
            self.dyn_map_arrays.append(
                torch.tensor([dyn_vocab2id.get(w) for w in unstructured_knowledge], requires_grad=False))

            vocab_map = []
            vocab_overlap = []
            for i in range(len(dyn_id2vocab)):
                vocab_map.append(self.vocab2id.get(dyn_id2vocab[i], self.vocab2id[UNK_WORD]))  # 用背景知识用词表来表示
                if dyn_id2vocab[i] in self.vocab2id:
                    vocab_overlap.append(0.)
                else:
                    vocab_overlap.append(1.)
            self.vocab_map_arrays.append(torch.tensor(vocab_map, requires_grad=False))  # 同上
            self.vocab_overlap_arrays.append(
                torch.tensor(vocab_overlap, requires_grad=False))  # 如果背景知识词在词表存在为0，否则为1

            # e_id = sample['id']
            # self.example_id.append(e_id)

            self.ids.append(id)
            self.id_arrays.append(torch.tensor([id]).long())

            self.context_id.append(sample['context_id'])
            self.query_id.append(sample['query_id'])

            if len(self.contexts) >= self.n:
                break
        self.len = len(self.contexts)
        print('full data size: ', self.len)

    def __getitem__(self, index):
        return [self.id_arrays[index], self.context_arrays[index], self.unstructured_knowledge_arrays[index],
                self.response_arrays[index], self.dyn_response_arrays[index], self.dyn_map_arrays[index],
                self.vocab_map_arrays[index], self.vocab_overlap_arrays[index],
                (self.ids[index], self.dyn_id2vocabs[index]), (self.ids[index], self.dyn_vocab2ids[index]),
                self.selections[index], self.ref_start_arrays[index], self.ref_end_arrays[index]]

    def __len__(self):
        return self.len

    def input(self, id):
        return self.contexts[id]

    def output(self, id):
        return self.responses[id]

    def background(self, id):
        return self.unstructured_knowledges[id]
    
    def c_id(self, id):
        return self.context_id[id]
    
    def q_id(self, id):
        return self.query_id[id]


def collate_fn(data):
    id_a, context_a, unstructured_knowledge_a, response_a, dyn_response_a, dyn_map, vocab_map, vocab_overlap, dyn_id2vocab, dyn_vocab2id, selection, ref_start, ref_end = zip(
        *data)

    return {'id': torch.cat(id_a),
            'context': pad_sequence(context_a, batch_first=True),
            'response': pad_sequence(response_a, batch_first=True),
            'unstructured_knowledge': pad_sequence(unstructured_knowledge_a, batch_first=True),
            'dyn_response': pad_sequence(dyn_response_a, batch_first=True),
            'dyn_map': pad_sequence(dyn_map, batch_first=True),
            'vocab_map': pad_sequence(vocab_map, batch_first=True),
            'vocab_overlap': pad_sequence(vocab_overlap, batch_first=True, padding_value=1.),
            'dyn_id2vocab': dict(dyn_id2vocab),
            'dyn_vocab2id': dict(dyn_vocab2id),
            'selection': torch.cat(selection),
            'ref_start': torch.cat(ref_start),
            'ref_end': torch.cat(ref_end)}
