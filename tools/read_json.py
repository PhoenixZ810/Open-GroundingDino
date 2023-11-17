import json
import argparse
import os
from multiprocessing import Pool
from tqdm import tqdm
import random
from functools import partial

parser = argparse.ArgumentParser(description="Read json")
parser.add_argument('file', type=str)
parser.add_argument('-p', '--process', type=int, default=16)
args = parser.parse_args()


def multi_judge_json(json):
    false_file = []
    # v1_file = []
    # v2_file = []
    with Pool(processes=args.process) as pool:
        iters = pool.imap(func=judge_json, iterable=json['images'])
        for iter in tqdm(iters, total=len(json['images'])):
            if iter[1] == 'false':
                print(iter[0])
                false_file.append(iter[0])
            # if iter[1] == 'v1':
            #     v1_file.append(iter[0])
            # if iter[1] == 'v2':
            #     v2_file.append(iter[0])
    # print('v1 num='+str(len(v1_file)))
    # print('v2 num='+str(len(v2_file)))


def judge_json(j):
    # a, b, c, d = j['file_name'].split('/')
    image_dir = os.path.join(
        '/mnt/workspace/zhaoxiangyu/data/objects365v1_processed/data/train', j['file_name']
    )
    # 找出不存在的图片
    if os.path.exists(image_dir):
        return [j, 'pass']
    else:
        return [j, 'false']

    # 找出o365v2不存在图片对应的anno
    # if f'{c}/{d}' in [
    #     'patch16/objects365_v2_00908726.jpg',
    #     'patch6/objects365_v1_00320532.jpg',
    #     'patch6/objects365_v1_00320534.jpg',
    # ]:
    #     return [j, 'no exist']

    # 找出所有v1
    # if b == 'v1':
    #     return [f'{b}/{c}/{d}', 'v1']
    # elif b == 'v2':
    #     return [f'{b}/{c}/{d}', 'v2']


def multi_judge_jsonl(jsonl):
    false_file = []
    pool = Pool(processes=args.process)
    with pool as pool:
        iters = pool.imap(func=judge_jsonl, iterable=jsonl)
        for iter in tqdm(iters):
            if iter[1] == 'false':
                print(iter[0] + 'not exist')
                false_file.append(iter[0])


def judge_jsonl(jsonl):
    j = json.loads(jsonl)
    '''extract image from jsonl'''
    rel_path = os.path.join(j['filename'][0:5], j['filename'])
    absolute_path = os.path.join(
        '/mnt/workspace/zhaoxiangyu/open-groundingdino/data/grit_processed/images', rel_path
    )
    os.system('cp ' + absolute_path + ' data/grit_275/images/')
    return [j['filename'], 'pass']
    '''check image exist'''
    # a, b, c, d = j['filename'].split('/')
    # if f'{c}/{d}' in [
    #     'patch16/objects365_v2_00908726.jpg',
    #     'patch6/objects365_v1_00320532.jpg',
    #     'patch6/objects365_v1_00320534.jpg',
    # ]:
    #     return [f'{c}/{d}', 'false']
    # else:
    #     return [j, 'pass']


def random_sample(json):
    images = json['images']
    samples = random.sample(images, 100)
    samples_id = [sample['id'] for sample in samples]
    fun = partial(search, target=samples_id)
    with Pool(args.process) as pool:
        iters = pool.imap(func=fun, iterable=json['annotations'])
        for iter in tqdm(iters, total=len(json['annotations'])):
            pass


def search(j, target):
    file_dir = []
    if j[id] in target:
        file_dir.append(j)


def lower_json(json_str):
    # 将json字符串转换为字典对象
    json_dict = json.load(json_str)
    # 遍历字典中的每个键值对
    for key, value in json_dict.items():
        # 如果值是一个字符串类型，就将其转换为小写，并替换原来的值
        if isinstance(value, str):
            value = value.lower()
            json_dict[key] = value
    # 将字典对象转换回json字符串，并返回
    json_lower = json.dumps(json_dict)
    with open('data/odvg/o365_label_map_lower.json', 'w') as f:
        f.write(json_lower)


def generate_o365v1_labelmap(j):
    id_sort = sorted(j['categories'], key=lambda x: x['id'])
    id_list = {}
    for dic in id_sort:
        id_list[dic['id'] - 1] = dic['name']
    js = json.dumps(id_list)
    with open('data/odvg/o365_label_map_lower.json', 'w') as f:
        f.write(js)


def main(args):
    with open(args.file, 'r') as f:
        print('loading...')
        # #lower json
        # lower_json(f)
        # #judge jsonl
        # multi_judge_jsonl(f)
        # #judge json
        j = json.load(f)
        # generate_o365v1_labelmap(j)
        import pdb
        pdb.set_trace()
        # print('loading comlete')
        # multi_judge_json(j)


if __name__ == '__main__':
    main(args)
