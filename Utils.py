import random
import time
import nltk
from Constants import *
from torch.distributions.categorical import *
from modules.Utils import *
from transformers import BertTokenizer


def get_ms():
    return time.time() * 1000


def init_seed(seed=None):
    if seed is None:
        seed = int(get_ms() // 1000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def importance_sampling(prob, topk):
    m = Categorical(logits=prob)
    indices = m.sample((topk,)).transpose(0, 1)  # batch, topk

    values = prob.gather(1, indices)
    return values, indices


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask = (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    if torch.cuda.is_available():
        mask = mask.cuda()
    return mask


def start_end_mask(starts, ends, max_len):
    batch_size = len(starts)
    mask = torch.arange(1, max_len + 1)
    if torch.cuda.is_available():
        mask = mask.cuda()
    mask = mask.unsqueeze(0).expand(batch_size, -1)
    mask1 = mask >= starts.unsqueeze(1).expand_as(mask)
    mask2 = mask <= ends.unsqueeze(1).expand_as(mask)
    mask = (mask1 * mask2)
    return mask


def decode_to_end(model, data, vocab2id, max_target_length=None, schedule_rate=1, softmax=False, encode_outputs=None,
                  init_decoder_states=None, tgt=None, init_feedback_states=None):
    # if tgt is None:
    #     tgt = data['output']
    # 查看回复
    # print('train content: \n', tgt)
    batch_size = len(data['id'])
    if max_target_length is None:
        max_target_length = tgt.size(1)
    if encode_outputs is None:
        encode_outputs = model.encode(data)
    if init_decoder_states is None:
        init_decoder_states = model.init_decoder_states(data, encode_outputs)  # [batch_size, 1, hidden_size]
        # print('init_decoder_states: ', init_decoder_states.size())
    # feedback初始化
    if init_feedback_states is None:
        feedback_outputs = model.init_feedback_states(data, encode_outputs, init_decoder_states)

    decoder_input = new_tensor([vocab2id[BOS_WORD]] * batch_size, requires_grad=False)  # 当前的前一个词

    prob = torch.ones((batch_size,)) * schedule_rate
    if torch.cuda.is_available():
        prob = prob.cuda()

    all_gen_outputs = list()  # 存储pk, pv mix后的p
    all_decode_outputs = [dict({'state': init_decoder_states})]  # 存储每个解码过程的pk, pv, state
    all_feedback_states = list()  # 保存所有的feedback state

    for t in range(max_target_length):

        if t != 0:
            all_decode_inputs = tgt[:, :t]  # 左闭右开，取到的是t之前所有的词
            # 待完成：model.decoder_to_encoder()、完成feedback.forward()、修改model.decode()、测试、控制feedback维度
            # feedback，输入为encoder输出（segment）和当前生成词前面的真值
            feedback_outputs = model.decoder_to_encoder(data, encode_outputs,
                                                        all_decode_inputs)  # all_decode_inputs用于GRU
        all_feedback_states.append(feedback_outputs)
        # decoder_outputs, decoder_states, ...
        # 生成每个解码过程的pk, pv, state
        decode_outputs = model.decode(
            data, decoder_input, encode_outputs, all_decode_outputs[-1], feedback_outputs
        )
        # 生成pk, pv mix后的p
        output = model.generate(data, encode_outputs, decode_outputs, softmax=softmax)

        all_decode_outputs.append(decode_outputs)
        all_gen_outputs.append(output)

        # 查看维度
        # print('*' * 20, 'Decoder模块', '*' * 20)
        # print('decode_outputs: ', decode_outputs)
        # print('output: ', output)
        # print('all_gen_outputs:', all_gen_outputs)
        # print('all_decode_outputs: ', all_decode_outputs)

        if schedule_rate >= 1:
            decoder_input = tgt[:, t]  # decoder前一个词使用的是真实回复的前一个词
        elif schedule_rate <= 0:
            probs, ids = model.to_word(data, output, 1)
            decoder_input = model.generation_to_decoder_input(data, ids[:, 0])
        else:
            probs, ids = model.to_word(data, output, 1)
            indices = model.generation_to_decoder_input(data, ids[:, 0])

            draws = torch.bernoulli(prob).long()
            decoder_input = tgt[:, t] * draws + indices * (1 - draws)

    # all_gen_outputs = torch.cat(all_gen_outputs, dim=0).transpose(0, 1).contiguous()

    return encode_outputs, init_decoder_states, all_decode_outputs, all_gen_outputs, all_feedback_states


def randomk(gen_output, k=5, PAD=None, BOS=None, UNK=None):
    if PAD is not None:
        gen_output[:, PAD] = -float('inf')
    if BOS is not None:
        gen_output[:, BOS] = -float('inf')
    if UNK is not None:
        gen_output[:, UNK] = -float('inf')
    values, indices = importance_sampling(gen_output, k)
    # words=[[tgt_id2vocab[id.item()] for id in one] for one in indices]
    return values, indices


def topk(gen_output, k=5, PAD=None, BOS=None, UNK=None):
    if PAD is not None:
        gen_output[:, PAD] = 0
    if BOS is not None:
        gen_output[:, BOS] = 0
    if UNK is not None:
        gen_output[:, UNK] = 0
    if k > 1:
        values, indices = torch.topk(gen_output, k, dim=1, largest=True,
                                     sorted=True, out=None)
    else:
        values, indices = torch.max(gen_output, dim=1, keepdim=True)
    return values, indices


def copy_topk(gen_output, vocab_map, vocab_overlap, k=5, PAD=None, BOS=None, UNK=None):
    vocab = gen_output[:, :vocab_map.size(-1)]
    dy_vocab = gen_output[:, vocab_map.size(-1):]

    vocab = vocab + torch.bmm(dy_vocab.unsqueeze(1), vocab_map).squeeze(1)
    dy_vocab = dy_vocab * vocab_overlap

    gen_output = torch.cat([vocab, dy_vocab], dim=-1)
    return topk(gen_output, k, PAD=PAD, BOS=BOS, UNK=UNK)


def remove_duplicate_once(sents, n=3):
    changed = False
    for b in range(len(sents)):
        sent = sents[b]
        if len(sent) <= n:
            continue

        for i in range(len(sent) - n):
            index = len(sent) - i - n
            if all(elem in sent[:index] for elem in sent[index:]):
                sents[b] = sent[:index]
                changed = True
                break
    return changed


def remove_duplicate(sents, n=3):
    changed = remove_duplicate_once(sents, n)
    while changed:
        changed = remove_duplicate_once(sents, n)


def to_sentence(batch_indices, id2vocab):
    batch_size = len(batch_indices)
    summ = list()
    for i in range(batch_size):
        indexes = batch_indices[i]
        text_summ2 = []
        for index in indexes:
            index = index.item()
            w = id2vocab[index]
            if w == BOS_WORD or w == PAD_WORD:
                continue
            if w == EOS_WORD:
                break
            text_summ2.append(w)
        if len(text_summ2) == 0:
            text_summ2.append(UNK_WORD)
        summ.append(text_summ2)
    return summ


def to_copy_sentence(data, batch_indices, id2vocab, dyn_id2vocab_map):
    ids = data['id']
    batch_size = len(batch_indices)
    summ = list()
    for i in range(batch_size):
        indexes = batch_indices[i]
        text_summ2 = []
        dyn_id2vocab = dyn_id2vocab_map[ids[i].item()]
        for index in indexes:
            index = index.item()
            if index < len(id2vocab):
                w = id2vocab[index]
            elif index - len(id2vocab) in dyn_id2vocab:
                w = dyn_id2vocab[index - len(id2vocab)]
            else:
                w = PAD_WORD

            if w == BOS_WORD or w == PAD_WORD:
                continue

            if w == EOS_WORD:
                break

            text_summ2.append(w)

        if len(text_summ2) == 0:
            text_summ2.append(UNK_WORD)

        summ.append(text_summ2)
    return summ


def universal_sentence_embedding(sentences, mask, sqrt=True):
    '''
    :param sentences: [batch_size, seq_len, hidden_size]
    :param mask: [batch_size, seq_len]
    :param sqrt:
    :return: [batch_size, hidden_size]
    '''
    # need to mask out the padded chars
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = (mask.sum(dim=1).view(-1, 1).float())
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


def bert_tokenizer():
    t = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True)  # do_lower_case Whether to lower case the input.
    return t.tokenize, t.vocab, t.ids_to_tokens


def bert_detokenizer():
    def detokenizer(tokens):
        return ' '.join(tokens).replace(' ##', '').strip()

    return detokenizer


def nltk_tokenizer():
    def tokenizer(sent):
        return nltk.word_tokenize(sent.lower())

    return tokenizer


def nltk_detokenizer():
    def detokenizer(tokens):
        return ' '.join(tokens)

    return detokenizer
