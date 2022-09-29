import nltk
from collections import Counter

base_path = 'dataset/wizard_of_wikipedia/'
vocab_list = []
# knowledge
for line in open(base_path + 'wizard_of_wikipedia.passage', 'r', encoding='utf8'):
    cur = line.split('\t')[-1].split('__knowledge__')[-1].strip().lower()
    vocab_list += nltk.word_tokenize(cur)
print('knowledge finish!')

# context
for line in open(base_path + 'wizard_of_wikipedia.query', 'r', encoding='utf8'):
    cur = line.split('\t')[-1].strip().lower()
    vocab_list += nltk.word_tokenize(cur)
print('context finish!')

# response
for line in open(base_path + 'wizard_of_wikipedia.answer', 'r', encoding='utf8'):
    cur = line.split('\t')[-1].strip().lower()
    vocab_list += nltk.word_tokenize(cur)
print('response finish!')

# write
dic = dict(Counter(vocab_list))
with open(base_path + 'wow_input_output.vocab', 'a+', encoding='utf8') as f:
    for i in dic.items():
        f.write(i[0] + '\t' + str(i[1]) + '\n')
print('count finish!')
