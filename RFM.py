from EncDecModel import *
from modules.BilinearAttention import *
from torch.distributions.categorical import Categorical
from modules.Highway import *
from data.Utils import *


class GenEncoder(nn.Module):
    def __init__(self, n, src_vocab_size, embedding_size, hidden_size, emb_matrix=None):
        super(GenEncoder, self).__init__()
        self.n = n

        if emb_matrix is None:
            self.c_embedding = nn.ModuleList(
                [nn.Embedding(src_vocab_size, embedding_size, padding_idx=0) for i in range(n)])
        else:
            self.c_embedding = nn.ModuleList(
                [create_emb_layer(emb_matrix) for i in range(n)])  # [1, all_word_num, emb_dim]
        self.c_encs = nn.ModuleList([nn.GRU(embedding_size, int(hidden_size / 2), num_layers=1, bidirectional=True,
                                            batch_first=True) if i == 0 else nn.GRU(embedding_size + hidden_size,
                                                                                    int(hidden_size / 2), num_layers=1,
                                                                                    bidirectional=True,
                                                                                    batch_first=True) for i in
                                     range(n)])

    def forward(self, c):  # [batch_size, word_num]    word_num: knowledge_len(300) or context_len(65)
        c_outputs = []
        c_states = []

        c_mask = c.ne(0).detach()  # [batch_size, word_num]
        c_lengths = c_mask.sum(dim=1).detach().cpu()  # [batch_size]

        c_emb = F.dropout(self.c_embedding[0](c), training=self.training)
        c_enc_output = c_emb  # [batch_size, word_num, emb_dim]   word_num: knowledge_len(300) or context_len(65)

        for i in range(self.n):  # self.n == 1
            if i > 0:
                c_enc_output = torch.cat([c_enc_output, F.dropout(self.c_embedding[i](c), training=self.training)],
                                         dim=-1)
            c_enc_output, c_state = gru_forward(self.c_encs[i], c_enc_output, c_lengths)

            c_outputs.append(c_enc_output.unsqueeze(1))
            c_states.append(c_state.view(c_state.size(0), -1).unsqueeze(1))

        return torch.cat(c_outputs, dim=1), torch.cat(c_states,
                                                      dim=1)  # [batch_size, 1, word_num, hidden_size] [batch_size, 1, hidden_size]


class KnowledgeSelector(nn.Module):
    def __init__(self, hidden_size, min_window_size=5, n_windows=4):
        super(KnowledgeSelector, self).__init__()
        self.min_window_size = min_window_size
        self.n_windows = n_windows

        self.b_highway = Highway(hidden_size * 2, hidden_size * 2, num_layers=2)
        self.c_highway = Highway(hidden_size * 2, hidden_size * 2, num_layers=2)
        self.match_attn = BilinearAttention(query_size=hidden_size * 2, key_size=hidden_size * 2,
                                            hidden_size=hidden_size * 2)
        self.area_attn = BilinearAttention(query_size=hidden_size, key_size=hidden_size, hidden_size=hidden_size)

    def match(self, b_enc_output, c_enc_output, c_state, b_mask, c_mask):
        b_enc_output = self.b_highway(torch.cat([b_enc_output, c_state.expand(-1, b_enc_output.size(1), -1)], dim=-1))
        c_enc_output = self.c_highway(torch.cat([c_enc_output, c_state.expand(-1, c_enc_output.size(1), -1)], dim=-1))

        matching = self.match_attn.matching(b_enc_output, c_enc_output)

        matching = matching.masked_fill(~c_mask.unsqueeze(1), -float('inf'))
        matching = matching.masked_fill(~b_mask.unsqueeze(2), 0)

        score = matching.max(dim=-1)[0]

        return score

    def segments(self, b_enc_output, b_score, c_state):
        window_size = self.min_window_size
        bs = list()
        ss = list()
        for i in range(self.n_windows):
            b = b_enc_output.unfold(1, window_size, self.min_window_size)
            b = b.transpose(2, 3).contiguous()
            b = self.area_attn(c_state.unsqueeze(1), b, b)[0].squeeze(2)
            bs.append(b)

            s = b_score.unfold(1, window_size, self.min_window_size)
            s = s.sum(dim=-1)
            ss.append(s)

            window_size += self.min_window_size
        return torch.cat(bs, dim=1), torch.cat(ss, dim=1)

    def forward(self, b_enc_output, c_enc_output, c_state, b_mask, c_mask):
        b_score = self.match(b_enc_output, c_enc_output, c_state, b_mask, c_mask)  # [batch_size, knowledge_len]
        segments, s_score = self.segments(b_enc_output, b_score, c_state)

        s_score = F.softmax(s_score, dim=-1)  # [batch_size, knowledge_len/4]

        segments = torch.bmm(s_score.unsqueeze(1), segments)  # [batch_size, 1, hidden_size]

        return segments, s_score, b_score
        # [batch_size, 1, hidden_size], [batch_size, knowledge_len/window_size(4)], [batch_size, knowledge_len]


class CopyGenerator(nn.Module):
    def __init__(self, embedding_size, hidden_size, knowledge_len):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(knowledge_len, hidden_size)
        self.cat_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.b_attn = BilinearAttention(query_size=embedding_size + hidden_size * 2, key_size=hidden_size,
                                        hidden_size=hidden_size)

    def forward(self, p, word, state, feedback_states, segment, b_enc_output, c_enc_output, b_mask, c_mask):
        feedback_states = self.linear(feedback_states)
        segment_mix = self.cat_linear(torch.cat([feedback_states, segment], dim=-1))
        p = self.b_attn.score(torch.cat([word, state, segment_mix], dim=-1), b_enc_output,
                              mask=b_mask.unsqueeze(1)).squeeze(1)
        return p


class VocabGenerator(nn.Module):
    def __init__(self, embedding_size, hidden_size, knowledge_len, vocab_size):
        super(VocabGenerator, self).__init__()
        self.linear = nn.Linear(knowledge_len, hidden_size)
        self.cat_linear = nn.Linear(hidden_size * 2, hidden_size)

        self.c_attn = BilinearAttention(query_size=embedding_size + hidden_size * 2, key_size=hidden_size,
                                        hidden_size=hidden_size)
        self.b_attn = BilinearAttention(query_size=embedding_size + hidden_size * 2, key_size=hidden_size,
                                        hidden_size=hidden_size)

        self.readout = nn.Linear(embedding_size + 4 * hidden_size, hidden_size)
        self.generator = nn.Linear(hidden_size, vocab_size)

    def forward(self, p, word, state, feedback_states, segment, b_enc_output, c_enc_output, b_mask, c_mask):
        feedback_states = self.linear(feedback_states)
        segment_mix = self.cat_linear(torch.cat([feedback_states, segment], dim=-1))

        c_output, _ = self.c_attn(torch.cat([word, state, segment_mix], dim=-1), c_enc_output, c_enc_output,
                                  mask=c_mask.unsqueeze(1))
        c_output = c_output.squeeze(1)

        b_output, _ = self.b_attn(torch.cat([word, state, segment_mix], dim=-1), b_enc_output, b_enc_output,
                                  mask=b_mask.unsqueeze(1))
        b_output = b_output.squeeze(1)

        concat_output = torch.cat((word.squeeze(1), state.squeeze(1), segment_mix.squeeze(1), c_output, b_output),
                                  dim=-1)

        feature_output = self.readout(concat_output)

        p = F.softmax(self.generator(feature_output), dim=-1)

        return p


class StateTracker(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(StateTracker, self).__init__()

        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)

    def initialize(self, segment, state):
        return self.linear(torch.cat([state, segment], dim=-1))

    def forward(self, word, state):
        return self.gru(word, state.transpose(0, 1))[1].transpose(0, 1)


class Mixturer(nn.Module):
    def __init__(self, hidden_size):
        super(Mixturer, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)

    def forward(self, state, dists1, dists2, dyn_map):
        p_k_v = torch.sigmoid(self.linear1(state.squeeze(1)))

        dists2 = torch.bmm(dists2.unsqueeze(1), dyn_map).squeeze(1)

        dist = torch.cat([p_k_v * dists1, (1. - p_k_v) * dists2], dim=-1)

        return dist


# feedback mechanism
class Feedback(nn.Module):
    def __init__(self, embedding_size, knowledge_len, context_len, hidden_size):
        super(Feedback, self).__init__()

        self.b_linear = nn.Linear(hidden_size * 2, knowledge_len)
        self.b_gru = nn.GRU(embedding_size, knowledge_len, num_layers=1, bidirectional=False, batch_first=True)

    def initialize(self, segment, state):
        b_init = self.b_linear(torch.cat([state, segment], dim=-1))  # [batch_size, 1, knowledge_len]
        return b_init

    def forward(self, word_emb, b_states):
        b_state = self.b_gru(word_emb, b_states.transpose(0, 1))[1].transpose(0, 1)  # [batch_size, 1, knowledge_len]
        return b_state


class Criterion(object):
    def __init__(self, tgt_vocab_size, eps=1e-10):
        super(Criterion, self).__init__()
        self.eps = eps
        self.offset = tgt_vocab_size

    def __call__(self, gen_output, response, dyn_response, UNK, reduction='mean'):
        dyn_not_pad = dyn_response.ne(0).float()
        v_not_unk = response.ne(UNK).float()
        v_not_pad = response.ne(0).float()

        if len(gen_output.size()) > 2:
            gen_output = gen_output.view(-1, gen_output.size(-1))

        p_dyn = gen_output.gather(1, dyn_response.view(-1, 1) + self.offset).view(-1)
        p_dyn = p_dyn.mul(dyn_not_pad.view(-1))

        p_v = gen_output.gather(1, response.view(-1, 1)).view(-1)
        p_v = p_v.mul(v_not_unk.view(-1))

        p = p_dyn + p_v + self.eps
        p = p.log()

        loss = -p.mul(v_not_pad.view(-1))
        if reduction == 'mean':
            return loss.sum() / v_not_pad.sum()
        elif reduction == 'none':
            return loss.view(response.size())


class RFM(EncDecModel):
    def __init__(self, min_window_size, num_windows, embedding_size, knowledge_len, context_len, hidden_size, vocab2id,
                 id2vocab, max_dec_len,
                 beam_width, emb_matrix=None, eps=1e-10):
        super(RFM, self).__init__(vocab2id=vocab2id, max_dec_len=max_dec_len, beam_width=beam_width, eps=eps)
        self.vocab_size = len(vocab2id)
        self.vocab2id = vocab2id
        self.id2vocab = id2vocab

        self.b_encoder = GenEncoder(1, self.vocab_size, embedding_size, hidden_size, emb_matrix=emb_matrix)
        self.c_encoder = GenEncoder(1, self.vocab_size, embedding_size, hidden_size, emb_matrix=emb_matrix)

        if emb_matrix is None:
            self.embedding = nn.Embedding(self.vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding = create_emb_layer(emb_matrix)  # 得到glove embedding

        if emb_matrix is None:
            self.feedback_embedding = nn.Embedding(self.vocab_size, embedding_size, padding_idx=0)
        else:
            self.feedback_embedding = create_emb_layer(emb_matrix)  # 得到glove embedding

        self.state_tracker = StateTracker(embedding_size, hidden_size)

        self.k_selector = KnowledgeSelector(hidden_size, min_window_size=min_window_size, n_windows=num_windows)

        self.c_generator = CopyGenerator(embedding_size, hidden_size, knowledge_len)
        self.v_generator = VocabGenerator(embedding_size, hidden_size, knowledge_len, self.vocab_size)

        self.mixture = Mixturer(hidden_size)

        self.criterion = Criterion(self.vocab_size)

        self.feedback = Feedback(embedding_size, knowledge_len, context_len, hidden_size)

        self.segment_linear = nn.Linear(int(knowledge_len / min_window_size + knowledge_len), hidden_size)

    def encode(self, data):
        b_enc_outputs, b_states = self.b_encoder(
            data['unstructured_knowledge'])  # [batch_size, 1, knowledge_len, hidden_size], [batch_size, 1, hidden_size]
        c_enc_outputs, c_states = self.c_encoder(
            data['context'])  # [batch_size, 1, context_len, hidden_size], [batch_size, 1, hidden_size]
        b_enc_output = b_enc_outputs[:, -1]  # [batch_size, knowledge_len, hidden_size]
        b_state = b_states[:, -1].unsqueeze(1)  # [batch_size, 1, hidden_size]
        c_enc_output = c_enc_outputs[:, -1]  # [batch_size, context_len, hidden_size]
        c_state = c_states[:, -1].unsqueeze(1)  # [batch_size, 1, hidden_size]

        _, p_s, p_g = self.k_selector(b_enc_output, c_enc_output, c_state, data['unstructured_knowledge'].ne(0),
                                      data['context'].ne(0))
        # [batch_size, 1, hidden_size], [batch_size, knowledge_len/window_size(4)], [batch_size, knowledge_len]

        s_g = torch.cat((p_s.unsqueeze(1), p_g.unsqueeze(1)), dim=-1)
        segment = self.segment_linear(s_g)

        return {'b_enc_output': b_enc_output, 'b_state': b_state, 'c_enc_output': c_enc_output, 'c_state': c_state,
                'segment': segment, 'p_s': p_s, 'p_g': p_g}

    def init_decoder_states(self, data, encode_outputs):
        return self.state_tracker.initialize(encode_outputs['segment'], encode_outputs['c_state'])

    def init_feedback_states(self, data, encode_outputs, init_decoder_states):
        return self.feedback.initialize(encode_outputs['segment'], init_decoder_states)

    def decode(self, data, previous_word, encode_outputs, previous_deocde_outputs, feedback_outputs):
        word_embedding = F.dropout(self.embedding(previous_word), training=self.training).unsqueeze(1)

        states = previous_deocde_outputs['state']
        states = self.state_tracker(word_embedding, states)  # [batch_size, 1, hidden_size]

        if 'p_k' in previous_deocde_outputs:
            p_k = previous_deocde_outputs['p_k']
            p_v = previous_deocde_outputs['p_v']
        else:
            p_k = None
            p_v = None

        p_k = self.c_generator(p_k, word_embedding, states, feedback_outputs, encode_outputs['segment'],
                               encode_outputs['b_enc_output'],
                               encode_outputs['c_enc_output'], data['unstructured_knowledge'].ne(0),
                               data['context'].ne(0))
        p_v = self.v_generator(p_v, word_embedding, states, feedback_outputs, encode_outputs['segment'],
                               encode_outputs['b_enc_output'],
                               encode_outputs['c_enc_output'], data['unstructured_knowledge'].ne(0),
                               data['context'].ne(0))

        return {'p_k': p_k, 'p_v': p_v, 'state': states}

    def generate(self, data, encode_outputs, decode_outputs, softmax=True):
        p = self.mixture(decode_outputs['state'], decode_outputs['p_v'], decode_outputs['p_k'],
                         data['dyn_map'])  # [batch_size, ]
        return {'p': p}

    def decoder_to_encoder(self, data, encode_outputs, gen_response):
        word_embedding = F.dropout(self.feedback_embedding(gen_response), training=self.training)
        p_g = encode_outputs['p_g'].unsqueeze(1)  # p_g [batch_size, 1, knowledge_len]
        init_feedback_states = torch.zeros_like(p_g)  # 第0个state为0，即h0=0    [batch_size, 1, knowledge_len]
        feedback_outputs = self.feedback(word_embedding, init_feedback_states)  # 最后一位state [batch_size, 1, hidden_size]
        response_matrix = torch.bmm(feedback_outputs.transpose(1, 2),
                                    p_g)  # response-aware weight matrix  [batch_size, knowledge_len, knowledge_len]
        attention_matrix = F.softmax(torch.bmm(p_g.transpose(1, 2), p_g),
                                     dim=-1)  # attention weight matrix   [batch_size, knowledge_len, knowledge_len]
        response_weight = torch.bmm(response_matrix,
                                    attention_matrix)  # response attention matrix  [batch_size, knowledge_len, knowledge_len]
        response_attention = torch.bmm(response_weight, p_g.transpose(1, 2)).transpose(1,
                                                                                       2)  # [batch_size, 1, knowledge_len]
        return response_attention

    def generation_to_decoder_input(self, data, indices):
        return indices.masked_fill(indices >= self.vocab_size, self.vocab2id[UNK_WORD])

    def to_word(self, data, gen_output, k=5, sampling=False):
        gen_output = gen_output['p']
        if not sampling:
            return copy_topk(gen_output, data['vocab_map'], data['vocab_overlap'], k=k, PAD=self.vocab2id[PAD_WORD],
                             UNK=self.vocab2id[UNK_WORD], BOS=self.vocab2id[BOS_WORD])
        else:
            return randomk(gen_output[:, :self.vocab_size], k=k, PAD=self.vocab2id[PAD_WORD],
                           UNK=self.vocab2id[UNK_WORD], BOS=self.vocab2id[BOS_WORD])

    def to_sentence(self, data, batch_indices):
        return to_copy_sentence(data, batch_indices, self.id2vocab, data['dyn_id2vocab'])

    def forward(self, data, method='mle_train'):
        data['dyn_map'] = build_map(data['dyn_map'])
        if 'train' in method:
            return self.do_train(data, type=method)
        elif method == 'test':
            data['vocab_map'] = build_map(data['vocab_map'], self.vocab_size)
            if self.beam_width == 1:
                return self.greedy(data)
            else:
                return self.beam(data)

    def do_train(self, data, type='mle_train'):
        encode_output, init_decoder_state, all_decode_output, all_gen_output, all_feedback_states = decode_to_end(self,
                                                                                                                  data,
                                                                                                                  self.vocab2id,
                                                                                                                  tgt=data['response'])
        loss = list()
        if 'mle' in type:
            p = torch.cat([p['p'].unsqueeze(1) for p in all_gen_output], dim=1)
            p = p.view(-1, p.size(-1))
            r_loss = self.criterion(p, data['response'], data['dyn_response'], self.vocab2id[UNK_WORD],
                                    reduction='mean').unsqueeze(0)
            loss += [r_loss]
        if 'mcc' in type:
            e1_loss = 1 - 0.1 * Categorical(probs=p[:, :self.vocab_size] + self.eps).entropy().mean().unsqueeze(0)
            e2_loss = 1 - 0.1 * Categorical(probs=p[:, self.vocab_size:] + self.eps).entropy().mean().unsqueeze(0)
            loss += [e1_loss, e2_loss]
        if 'ds' in type:
            k_loss = F.kl_div((encode_output['p_s'].squeeze(1) + self.eps).log(), data['selection'] + self.eps,
                              reduction='batchmean').unsqueeze(0)
            loss += [k_loss]
        return loss
