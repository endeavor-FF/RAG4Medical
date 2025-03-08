#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : parser.py
@Author : WangFeng
@Date   : 2025/3/7 17:57
@Desc   : 
"""
import fitz  # pymupdf
import copy
from dataclasses import dataclass, field, asdict
from typing import List, Dict
import re
import json
from tqdm import tqdm


@dataclass
class MedicineAttribute:
    cls: str = ""
    attr: str = ""
    attr_ctx: str = ""

    def is_empty(self) -> bool:
        return self.attr == ''

    def set_name(self, attr: str):
        self.attr = attr

    def set_ctx(self, ctx: str):
        self.attr_ctx = ctx

    def clear(self):
        self.attr = ''
        self.attr_ctx = ''


@dataclass
class Medicine:
    name: str = ""
    data: List[MedicineAttribute] = field(default_factory=list)

    def add_entry(self, entry: MedicineAttribute):
        """向 data 列表中添加一个字典。"""
        if isinstance(entry, MedicineAttribute):
            self.data.append(entry)
        else:
            raise ValueError("Entry must be a dictionary")

    def is_empty(self) -> bool:
        return self.name == ''

    def set_name(self, name: str):
        self.name = name

    def clear(self):
        self.name = ''
        self.data = []


@dataclass
class CoarseMedicine:
    name: str = ""
    common_info: str = ""
    sp_info: str = ""

    def is_empty(self) -> bool:
        return self.name == ''

    def set_common_info(self, info):
        self.common_info = info

    def set_sp_info(self, info):
        self.sp_info = info

    def set_name(self, name: str):
        self.name = name

    def clear(self):
        self.name, self.sp_info, self.common_info = '', '', ''


def parse_pdf(pdf_path, file):
    # 打开PDF文件
    wr = open(file, 'w', encoding='utf-8')
    doc = fitz.open(pdf_path)
    results = []
    common_ctx = ''
    sp_ctx = ''
    medicine_instance = CoarseMedicine()
    for page_num in tqdm(range(len(doc)), desc="Processing"):
        page = doc.load_page(page_num)  # 加载每一个页面
        blocks = page.get_text("dict")["blocks"]  # 提取文本块
        for idx, block in enumerate(blocks):
            if idx == 0 :
                continue  # 去掉页眉和页码
            if "lines" in block:  # 检查是否为文本块
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()  # 提取文本
                        font_size = int(span["size"])  # 字体大小
                        font_name = span["font"]  # 字体名称
                        if '•' in text and (font_size == 9 or font_size == 8):
                            break
                        if (font_name == 'Helvetica' and font_size == 8) or (
                                font_name == 'Times-Roman' and font_size == 11):
                            break

                        if font_size == 13 or font_size == 14:  # 药物品类
                            if medicine_instance.is_empty():  # 当前为空
                                medicine_instance.set_name(text)
                            else:
                                # 不为空
                                medicine_instance.set_sp_info(sp_ctx)
                                medicine_instance.set_common_info(common_ctx)
                                wr.writelines(json.dumps(asdict(medicine_instance), ensure_ascii=False) + '\n')
                                results.append(copy.deepcopy(medicine_instance))
                                medicine_instance.clear()
                                sp_ctx, common_ctx = '', ''
                            continue
                        if text == '饮片':
                            sp_ctx += text
                            continue
                        if sp_ctx != '':
                            sp_ctx += text
                        else:
                            common_ctx += text
    return results


if __name__ == "__main__":
    pdf_path = "/Users/fengfeng/Downloads/《中国药典》2020年版 一部-48-1947.pdf"
    output_path = "./coarse_preprocess.json"
    parsed_data = parse_pdf(pdf_path, output_path)
    #
    # for item in parsed_data:
    #     # print(f"文本: {item['文本']}")
    #     # print(f"字体大小: {item['字体大小']}")
    #     # print(f"字体名称: {item['字体名称']}")
    #     # print(f"颜色: {item['颜色']}")
    #     # print(f"位置: {item['位置']}")
    #     # print(f"是否为标题: {item['是否为标题']}")
    #     # print("-" * 50)
