#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : get_embedding.py
@Author : WangFeng
@Date   : 2025/3/8 11:17
@Desc   : 
"""
import pandas as pd
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import os
from typing import List


def sentence2input(sens, max_len=10):
    input_ids = []
    att_mask = []
    for sen in sens:
        tsen = tokenizer.tokenize(sen)
        tsen = tsen[:max_len]
        tsen = tokenizer.convert_tokens_to_ids(tsen)
        tsen = [101] + tsen + [102]
        att_mask.append([1] * len(tsen) + [0] * (max_len + 2 - len(tsen)))
        tsen = tsen + [0] * (max_len + 2 - len(tsen))
        input_ids.append(tsen)
    return torch.tensor(input_ids), torch.tensor(att_mask)


def query2vector(query):
    query = query
    # tokens = tokenizer(query, padding = True, truncation=True, return_tensors='pt',max_length = 512)
    tokens, att = sentence2input([query], 500)
    tokens = tokens.to(device)
    att = att.to(device)
    model.eval()
    with torch.no_grad():
        out = model(input_ids=tokens, attention_mask=att)[0][:, 0]
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        print(f'debug 编码output: {out}')
    return out[0].cpu().numpy().tolist()


def split_string_with_overlap(s, block_size=400, overlap_size=100) -> List[str]:
    n = len(s)
    if n <= block_size:
        return [s]

    blocks = []
    start = 0
    while start < n:
        end = start + block_size
        if end > n:
            end = n
        blocks.append(s[start:end])
        start += (block_size - overlap_size)

    return blocks


def get_embedding(args):
    dic_lst = [json.loads(i) for i in open(args.in_jsonl_path, 'r', encoding='utf-8')]
    writer = open(file=args.out_jsonl_path, mode='w', encoding='utf-8')
    data = []
    pbar = tqdm(total=len(dic_lst))
    global_ids = 0
    for item in dic_lst:
        for k, v in item.items():
            if k not in ['name', 'common_info']:
                for block in split_string_with_overlap(f'{k}:{v}'):
                    dic = {
                        'rid': global_ids,
                        'sentence': block,
                        'parent_doc': f'{k}:{v}',
                        'parent_name': item['name'],
                        'parent_common_info': item['common_info']
                    }

                    query_vec = query2vector(dic['sentence'].replace('\n', ''))
                    vector_str = " ".join([str(vec) for vec in query_vec])
                    vec_dic = {
                        'id': dic['rid'],
                        'vec': vector_str
                    }
                    data.append(vec_dic)
                    writer.writelines(json.dumps(dic, ensure_ascii=False) + '\n')
                    pbar.update(1)
                    global_ids += 1
    writer.close()
    vec_df = pd.DataFrame(data)
    vec_df.to_csv(path_or_buf=args.out_csv_path, index=False, header=None)
    print(
        f"原{len(dic_lst)}条数据,共编码{len(vec_df)}条数据\njsonl文件输出至:{args.out_jsonl_path}\ncsv向量文件输出至:{args.out_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl_path", type=str, default="../data_preprocess/preprocess.json",
                        help="path to knowledge json")
    parser.add_argument("--out_jsonl_path", type=str, default="./output/doc.txt",
                        help="path to knowledge jsonl")
    parser.add_argument("--out_csv_path", type=str, default="./output/doc_vector",
                        help="path to knowledge csv")
    parser.add_argument("--model_tokenizer_path", type=str,
                        default="/Users/fengfeng/program/enterprise_program/bge-large-zh-v1.5",
                        help="a model for embeddings")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer_path)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(args.model_tokenizer_path)
    model.to(device)
    get_embedding(args)
