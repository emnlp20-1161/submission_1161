import csv
import os
import re
import numpy as np
import datetime
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def get_free_gpu(mem_min=5012, max_num_gpu=4):

    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp')
    memory_available = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
    os.system('rm tmp')
    valid_gpu_idx = np.where(memory_available > mem_min)[0]
#     valid_gpu_idx = np.delete(valid_gpu_idx, np.where(valid_gpu_idx == 7)[0])
    rank_mem = np.argsort(-memory_available[valid_gpu_idx])
    valid_gpu_idx = valid_gpu_idx[rank_mem[:max_num_gpu]]
    for idx in valid_gpu_idx:
        print(f"Using GPU #{idx}, with {memory_available[idx]} MiB free memory")

    return valid_gpu_idx


def clean_str(string):

    string = re.sub(r"\\n", " ", string)
    string = re.sub(r"\\$", " $ ", string)
    string = re.sub(r"\\", " ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"<br />", " ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string


def ids2string(ids, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)):
    
    string = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids))
    string = re.sub(rf"\{tokenizer.pad_token}", " ", string)
    string = re.sub(rf"\{tokenizer.sep_token}", " ", string)
    string = re.sub(rf"\{tokenizer.cls_token}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    return string


def csv2txt(file_dir, file_name):

    csv_file = open(os.path.join(file_dir, file_name))
    csv_reader = csv.reader(csv_file, delimiter=',')

    docs = []
    labels  = []
    imdb_map = {"positive": 1, "negative": 2}
    for i, row in enumerate(csv_reader):
        if "agnews" in file_dir or "dbpedia" in file_dir or "amazon" in file_dir:
            doc = row[1] + '. ' + row[2]
        elif "imdb" in file_dir:
            if i == 0:
                continue
            assert len(row) == 2
            doc = row[0]
            row[0] = imdb_map[row[1]]
        docs.append(doc)
        labels.append(int(row[0]) - 1)

    out_file_name = file_name.split('.')[0]
    out_corpus = open(os.path.join(file_dir, f"{out_file_name}.txt"), 'w')
    out_label = open(os.path.join(file_dir, f"{out_file_name}_labels.txt"), 'w')
    for doc, label in zip(docs, labels):
        out_corpus.write(clean_str(doc) + '\n')
        out_label.write(str(label) + '\n')

    return


def tokenize(docs, labels, max_len=512, show_stats=False):

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    if show_stats:
        doc_len = []
        for doc in docs:
            inputs = tokenizer.encode(doc, add_special_tokens=True)
            doc_len.append(len(inputs))

        print(f"doc length max: {np.max(doc_len)}, avg: {np.mean(doc_len)}, std: {np.std(doc_len)}")
        trunc_frac = np.sum(np.array(doc_len) > max_len) / len(doc_len)
        print(f"truncated fraction: {trunc_frac}")

    inputs = []
    attention_masks = []

    for doc in docs:
        encoded_dict = tokenizer.encode_plus(doc, add_special_tokens=True, max_length=max_len, pad_to_max_length=True,
                                             return_attention_mask=True, return_tensors='pt')
        inputs.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    inputs = torch.cat(inputs, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return inputs, attention_masks, labels


def create_dataset(file_dir, train_file, label_file, loader_name, show_stats=False, max_len=512, sampler=RandomSampler, batch_size=64):

    loader_file = os.path.join(file_dir, loader_name)
    if os.path.exists(loader_file):
        inputs, attention_masks, labels = torch.load(loader_file)
    else:
        corpus = open(os.path.join(file_dir, train_file))
        docs = [doc.strip() for doc in corpus.readlines()]
        truth = open(os.path.join(file_dir, label_file))
        labels = [int(label.strip()) for label in truth.readlines()]
        inputs, attention_masks, labels = tokenize(docs, labels, max_len, show_stats)
        torch.save([inputs, attention_masks, labels], loader_file)

    dataset = TensorDataset(inputs, attention_masks, labels)
    print(f"{len(dataset)} samples")
    dataloader = DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)

    return dataloader


def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def show_params(model):

    params = list(model.named_parameters())
    print('The model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    return
