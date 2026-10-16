import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base_dir", default="/mnt/ws-jfs/dsh-jfs/lrz/verl", help="input file")
parser.add_argument("-c", "--ckpt_dir", default="", help="input file")
parser.add_argument("-l", "--logs_dir", default="", help="input file")
parser.add_argument("-w", "--wandb_dir", default="", help="input file")
args = parser.parse_args()

if args.ckpt_dir:
    args.wandb_dir = args.ckpt_dir.replace("/ckpts/", "/wandb_dir/")
    args.logs_dir = args.ckpt_dir.replace("/ckpts/", "/logs/")
elif args.logs_dir:
    args.ckpt_dir = args.logs_dir.replace("/logs/", "/ckpts/")
    args.wandb_dir = args.logs_dir.replace("/logs/", "/wandb_dir/")
elif args.wandb_dir:
    args.ckpt_dir = args.wandb_dir.replace("/wandb_dir/", "/ckpts/")
    args.logs_dir = args.wandb_dir.replace("/wandb_dir/", "/logs/")
else:
    raise ValueError

print(f"ckpt_dir: {args.ckpt_dir}")
print(f"logs_dir: {args.logs_dir}")
print(f"wandb_dir: {args.wandb_dir}")

try:
    shutil.rmtree(args.ckpt_dir)
except Exception as e:
    print(e)
try:
    shutil.rmtree(args.logs_dir)
except Exception as e:
    print(e)
try:
    shutil.rmtree(args.wandb_dir)
except Exception as e:
    print(e)

