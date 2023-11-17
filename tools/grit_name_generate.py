import json
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

def image_num(image_dir, all_images):
    images = [i for i in os.listdir(image_dir) if i.endswith(".jpg")]
    return [images, len(images)]

def write(name):
    return name

def create_image_list(root, process):
    all_images = []
    num = 0
    image_dirs = [
        os.path.join(root, f)
        for f in os.listdir(root)
        if os.path.isdir(os.path.join(root, f))
    ]

    func1 = partial(image_num, all_images=all_images)
    with Pool(processes=process) as pool:
        iter = pool.imap(func=func1, iterable=image_dirs)
        for iter in tqdm(iter, total=len(image_dirs)):
            all_images.extend(iter[0])
            num += iter[1]
    print(num)

    func2 = partial(write)
    with open("/mnt/workspace/zhaoxiangyu/data/grit_processed/image_list.txt", "w") as f:
        with Pool(processes=process) as pool:
            iter = pool.imap(func=func2, iterable=all_images)
            for iter in tqdm(iter, total=len(all_images)):
                f.write(iter + "\n")
def main():
    root = "/mnt/workspace/zhaoxiangyu/data/grit_processed/images"
    process=32
    create_image_list(root, process)

if __name__== "__main__":
    main()