import re
def load_predata(file, tokenizer):
    sentences = []
    relations = []
    e1_pos_s = []
    e1_pos_e = []
    e2_pos_s = []
    e2_pos_e = []

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            line = line.strip().lower().split()
            relations.append(int(line[0]))
            e1_pos_s.append(int(line[1]))  # (start_pos, end_pos)
            e1_pos_e.append(int(line[2]))  # (start_pos, end_pos)
            e2_pos_s.append(int(line[3]))  # (start_pos, end_pos)
            e2_pos_e.append(int(line[4]))  # (start_pos, end_pos)
            sentences.append(line[5:])
    # return sentences, relations, e1_pos, e2_pos
    return  sentences,e1_pos_s, e1_pos_e, e2_pos_s, e2_pos_e


def load_data(datapath, tokenizer):
    """
    :param datapath(str): 文件路径
    :param tokenizer: 分词器
    :return sentences(list):list of sentence, relation(list):
    """
    sentences, relations = [], []
    with open(datapath) as fp:
        lines = fp.readlines()
        for i in range(0, len(lines), 2):
            sentence_num = lines[i].split(" ")[0]
            pattern = re.compile('"(.*)"')
            sentence = pattern.findall(lines[i])[0]
            # sentence = clean_text(sentence)
            relation = lines[i + 1].strip().split('(')[0]
            tokens = tokenizer.tokenize(sentence)
            sentence = ''
            for i, token in enumerate(tokens):
                sentence = sentence + ' ' + token
            relations.append(relation)
            sentences.append(sentence)
    return relations


def jointoken(tokens):
    ret = []
    for t in tokens:
        t = t.strip().split()
        t = " ".join(t)
        ret.append(t)
    return ret


def clean_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r'([.,!?"])', r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text
