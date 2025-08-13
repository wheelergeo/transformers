from src import qwen2_5_vl_divprune as qwen
from src.vit_patch16_224_tome import evaluate
from utils.plot import plot_performance


import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', 
                        type=str,
                        default='cuda:0',
                        help='Device to use for inference')
    parser.add_argument('--pretrained-model-path', 
                        type=str,
                        default='./checkpoints/vit-large-patch16-224',
                        help='Path to the pretrained model')
    parser.add_argument('--dataset-path', 
                        type=str,
                        default='./dataset/imagenet1k',
                        help='Path to the dataset')
    parser.add_argument('--results-save-path',
                        type=str,
                        default='./workdir/vit-large-patch16-224.perf',
                        help='Save results to JSON file')
    parser.add_argument('--tome-r',
                        type=int,
                        default=0,
                        help='Token merging\'s hyper-parameter')
    parser.add_argument('--use-safetensors',
                        default=False,
                        action='store_true',
                        help='Whether to use safetensors')
    args = parser.parse_args()

    # for i in range(0, 9):
    #     args.tome_r = i
    #     print(f"------------------tome_r: {i}------------------")
    #     evaluate(args)

    plot_performance(perf_dir="./workdir/vit-large-patch16-224.perf",
                     save_path="./workdir/vit-large-patch16-224.perf/performance.png")