# This script deletes all directories that start with patch in the current directory
import os
from multiprocessing import Pool
from tqdm import tqdm
import shutil
import argparse

parser = argparse.ArgumentParser('delete all referring file in dir')
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--process', type=int, default=8)
parser.add_argument('-m','--mode',type=str, default='file', help='choose dir or file')
args = parser.parse_args()

# Define a function to delete a directory
def delete_file(dir):
    if not os.path.isdir(dir):
      os.remove(dir)
    return f"Deleted {dir}"

def delete_dir(dir):
    if os.path.isdir(dir):  # check if it is a directory
      shutil.rmtree(dir)  # delete the directory
    return f"Deleted {dir}"

def main(args):
    dirs = [os.path.join(args.dir, dir) for dir in os.listdir(args.dir) if dir.endswith(".gz")]

    # Create a pool of processes
    if args.mode == 'dir':
      with Pool(processes=args.process) as pool:
          iters = pool.imap(func=delete_dir, iterable=dirs)
          for iter in tqdm(iters, total=len(dirs)):
              print(iter)
    elif args.mode == 'file':
      with Pool(processes=args.process) as pool:
          iters = pool.imap(func=delete_file, iterable=dirs)
          for iter in tqdm(iters, total=len(dirs)):
              print(iter)

if __name__=="__main__":
  main(args)