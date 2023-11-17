import jsonlines
from tqdm import tqdm
import random
import json
import os
from multiprocessing import Pool
from functools import partial
import emoji

import argparse

'''对caption中的标点符号进行清洗，中文标点转为英文标点，去掉句末的标点'''


def clean_span(span):
    span = span.rstrip()
    span = span.replace('"', "'").replace('\"', "'").replace('“', "'").replace('”', "'")
    span = span.replace('‘', "'").replace('’', "'").replace('–', "—")
    if span.endswith('/') or span.endswith('.'):
        span = span[:-1]
    return span


'''检查grit caption中是否有除了ASCII文本以外的其他元素（如.*/-emoji[CLS]等)，如果有返回False'''


def check_caption(cap):
    check_anno = cap["caption"].rstrip()[:-1]
    if not str.isascii(check_anno):
        return False
    # "The view is better from here 🦅 (Chouf" wtf??
    check_list = {"↙️", "-", ",", " ", "*", "/", "$", "[CLS]", "[SEP]", "?"}
    for ch in check_list:
        if ch in check_anno:
            return False
    if '.' in check_anno[:-1]:
        return False
    if emoji.emoji_count(check_anno):
        print(check_anno)
        return False
    return True


def get_regions(nc, anno):
    h = anno["height"]
    w = anno["width"]
    phrase = clean_span(anno["caption"][int(nc[0]) : int(nc[1])])
    bbox = [nc[2] * w, nc[3] * h, nc[4] * w, nc[5] * h]
    return {"bbox": bbox, "phrase": phrase}


# '''从所有图片中选择random_sampled张图片'''


def prepare_list(file_name: str, random_samples):
    with open(file_name, "r") as f:
        # metas = [line.strip() for line in f]
        metas = json.load(f)
    num_of_files = len(metas)
    print(num_of_files)
    metas = random.sample(metas, random_samples)
    num_of_files = len(metas)
    print("after sample:", num_of_files)
    return metas, num_of_files


def process_item(file, args):
    file_dir = file[0:5]
    json_name = file_dir + '.json'
    with open(os.path.join(os.path.dirname(args.root), 'annotations/', json_name)) as f:
        anno = json.load(f)
    for i in anno:
        if i['key'] == os.path.splitext(file)[0]:
            anno = i
            break
    anno = file
    # 若caption有除了ASCII文本以外的其他元素（.*/-emoji[CLS]等)，返回None
    if not check_caption(anno):
        return None
    noun_chunks = anno['noun_chunks']
    ref_exps = anno['ref_exps']
    regions = []
    random_num = random.random()
    # 随机选择采集名词或者短语，从sentence获得对应的名词或短语，以及region坐标
    if random_num > 0.5:
        for nc in noun_chunks:
            region = get_regions(nc, anno)
            if str.isascii(region["phrase"]):
                regions.append(region)
    else:
        for re in ref_exps:
            region = get_regions(re, anno)
            if str.isascii(region["phrase"]):
                regions.append(region)
    if len(regions) < args.min_phrase:
        return None

    odvg_anno = {
        "filename": file,
        "height": anno["height"],
        "width": anno["width"],
        "grounding": {"caption": clean_span(anno["caption"]), "regions": regions},
    }
    return odvg_anno


if __name__ == "__main__":
    # jsons = "/share_data/mllm/kosmos-2/GRIT-20M/anno/14m_anno.list"
    # root = "/share_data/mllm/kosmos-2/GRIT-20M/data"
    # output_name = "./girt_14m_odvg.jsonl"
    parser = argparse.ArgumentParser(description="GRIT2ODVG List.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--root", type=str, default="", help="Source image root")
    parser.add_argument("--output_file", type=str, default="./")
    parser.add_argument("--random_samples", type=int, default=2000)
    parser.add_argument("--chunk_or_ref", type=float, default=0.5)
    parser.add_argument("--min_phrase", type=int, default=6)
    parser.add_argument("--process_num", type=int, default=32, help="the number of processes")
    args = parser.parse_args()
    print(args)
    metas, metas_len = prepare_list(args.input_file, args.random_samples)
    odvg_anno = []
    func = partial(process_item, args=args)
    with Pool(processes=args.process_num) as pool:
        for result in tqdm(pool.imap(func=func, iterable=metas), total=len(metas)):
            odvg_anno.append(result)
    for meta in metas:
        result = func(meta)
    odvg_anno = list(filter(None, odvg_anno))
    json_name = os.path.join(args.output_file, f'girt_14m_odvg_{args.random_samples}.json')
    with jsonlines.open(json_name, mode="w") as fwriter:
        fwriter.write_all(odvg_anno)
