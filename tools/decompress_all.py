from multiprocessing import Pool
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description='untar all tar files')
parser.add_argument('dir', type=str)
parser.add_argument('mode', type=str, choices=['tar', 'tar.gz', 'zip'])
parser.add_argument('-p', '--process', type=int, default=8)
args = parser.parse_args()


def untar_gz(tar):
    tar_dir = tar.split('.')[-3]
    if os.path.exists(tar_dir) and os.path.isdir(tar_dir):
        os.system(f'rm -rf {tar_dir}')
    print(f'{tar} decompressing')
    tar_dirname = os.path.dirname(tar_dir)
    os.system(f'tar -zxf {tar} -C {tar_dirname}')
    return os.path.basename(tar)


def untar(tar):
    tar_dir = tar.split('.')[-2]
    if os.path.exists(tar_dir) and os.path.isdir(tar_dir):
        os.system(f'rm -rf {tar_dir}')
    print(f'{tar} decompressing')
    tar_dirname = os.path.dirname(tar_dir)
    os.system(f'tar -zxf {tar} -C {tar_dirname}')
    return os.path.basename(tar)


def unzip(zip):
    # zip_dir = zip.split('.')[-2]
    # if os.path.exists(zip_dir) and os.path.isdir(zip_dir):
    #     os.system(f'rm -rf {zip_dir}')
    print(f'{zip} decompressing')
    # zip_dirname = os.path.dirname(zip_dir)
    os.system(f'unzip {zip}')
    return os.path.basename(zip)


def main(args):
    # tar_gzs = [os.path.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith('.tar.gz')]
    # tars = [os.path.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith('.tar')]
    # zips = [os.path.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith('.zip')]
    if args.mode == 'tar.gz':
        fun = untar_gz
        file_list = [
            os.path.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith('.tar.gz')
        ]
    elif args.mode == 'tar':
        fun = untar
        file_list = [
            os.path.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith('.tar')
        ]
    elif args.mode == 'zip':
        fun = unzip
        file_list = [
            os.path.join(args.dir, file) for file in os.listdir(args.dir) if file.endswith('.zip')
        ]
    with Pool(processes=args.process) as pool:
        iters = pool.imap(func=fun, iterable=file_list)
        for iter in tqdm(iters, total=len(file_list)):
            print(f'{iter} finished')


if __name__ == '__main__':
    main(args)
