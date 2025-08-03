from src import qwen2_5_vl_divprune as qwen
from src import vit_base_patch16_384_tome as vit
from utils import plot_performance


import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-model-path', 
                        type=str,
                        default="/home/yrc/workspace/transformers/experiments/checkpoints/Qwen2.5-VL-7B-Instruct", 
                        help='Path to the pretrained model')
    parser.add_argument('--divprune-subset-ratio', 
                        type=float,
                        default=0.098,
                        help='Subset ratio for divPrune, default is 0.0')
    parser.add_argument('--image-path',
                        type=str,
                        default="/home/yrc/workspace/transformers/experiments/dataset/imagenet1k/val/n01440764/ILSVRC2012_val_00048969.JPEG",
                        # default="/home/yrc/workspace/transformers/experiments/dataset/imagenet1k/val/n01440764/ILSVRC2012_val_00000293.JPEG",
                        help='Path to the image for evaluation')
    parser.add_argument('--text-prompt',
                        type=str,
                        default="Tell me what I should pay attention to.",
                        help='Text prompt for the model')

    args = parser.parse_args()
    flops, params, inference_time, tps, inference_res = qwen.evaluate(args)
    
    print(f"Flops: {flops}, Params: {params}")
    print(f"Inference Time: {inference_time: .2f} s")
    print(f"TPS: {tps: .2f} tokens/s")
    print(f"Inference Output: {inference_res}")