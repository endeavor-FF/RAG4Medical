#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : doc_parser.py
@Author : WangFeng
@Date   : 2025/3/8 10:34
@Desc   : 
"""
from dataclasses import dataclass, asdict
from docx import Document
import json
import re
from collections import defaultdict


@dataclass
class CoarseMedicine:
    name: str = ""
    common_info: str = ""

    def is_empty(self) -> bool:
        return self.name == ''

    def set_common_info(self, info):
        self.common_info = info

    def set_name(self, name: str):
        self.name = name

    def clear(self):
        self.name, self.common_info = '', ''


def parse_document(document: str, target_file) -> list[CoarseMedicine]:
    doc = Document(document)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    medicines = []
    sections = text.split("###")  # 以"###"分割文档
    wr = open(target_file, 'w', encoding='utf-8')
    for section in sections:
        if not section.strip():  # 跳过空段落
            continue

        # 找到药品名称
        lines = section.strip().split('\n')
        name = lines[0].strip() if lines else ""

        # 合并剩余部分作为药品描述
        common_info = '\n'.join(lines[1:]).strip()

        # 创建药品实例并添加到列表
        medicine = CoarseMedicine()
        medicine.set_name(name)
        medicine.set_common_info(common_info)
        wr.writelines(json.dumps(asdict(medicine), ensure_ascii=False) + '\n')
        medicines.append(medicine)
    wr.close()
    return medicines


pattern = r"【(.*?)】(.*?)(?=【|$)"



def paser_again(js_ori: str, js_output: str):
    ori_dict_lst = [json.loads(i) for i in open(js_ori, 'r', encoding='utf-8')]
    wr = open(js_output, 'w', encoding='utf-8')
    for item in ori_dict_lst:
        matches = re.findall(pattern, item["common_info"], re.DOTALL)
        new_dict = defaultdict()
        if '丁香' in item['common_info']:
            pass
        for match in matches:
            key = f"{item['name']}的{match[0]}"
            value = match[1].strip()
            if key in new_dict:
                continue
            new_dict[key] = value
        new_dict['name'] = item['name']
        new_dict['common_info'] = item['common_info']
        wr.writelines(json.dumps(new_dict,ensure_ascii=False)+'\n')
    wr.close()


if __name__ == '__main__':
    doc_path = '/Users/fengfeng/Documents/parser_doc.docx'
    # parse_document(doc_path, './coarse_preprocess.json')
    paser_again('./coarse_preprocess.json', './preprocess.json')
