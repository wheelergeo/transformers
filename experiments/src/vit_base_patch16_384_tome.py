from transformers import pipeline
from transformers.models import ViTForImageClassification, ViTImageProcessor
from torchvision import datasets, transforms
from tqdm import tqdm
from thop import profile, clever_format


import os
import sys
import time
import torch
import argparse
import json


def inspect_labels():
    # generate dataloader
    data_transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = datasets.ImageNet(root=args.dataset_path, split='val', transform=data_transform)


@torch.no_grad()
def inference(model: torch.nn.Module,
              image_processor,
              data_loader: torch.utils.data.DataLoader, 
              device: str):
    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    sample_num = 0

    tqdm_loader = tqdm(data_loader, file=sys.stdout)
    for images, labels in tqdm_loader:
        images = image_processor(images, return_tensors="pt")
        images = {k: v.to(device) for k, v in images.items()}
        labels = labels.to(device)

        preds = model(**images).logits.argmax(dim=-1)

        accu_num += (preds == labels).sum()
        sample_num += labels.size(0)

        tqdm_loader.set_postfix(acc=f"{accu_num.item() / sample_num:.3f}")

    return accu_num.item() / sample_num


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not args.tome_r:
        os.environ["TOME_R"] = str(args.tome_r)
    
    # generate dataloader
    data_transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = datasets.ImageNet(root=args.dataset_path, split='val', transform=data_transform)
    total_images = len(test_dataset)

    batch_size = min(4, total_images)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, pin_memory=True,
                                              num_workers=nw, collate_fn=None)

    # create model
    model = ViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        use_safetensors=True,).to(device)
    image_processor = ViTImageProcessor.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
    )
    
    # warm-up GPU
    dummy_input = torch.randn(1, 3, 384, 384).to(device)
    # calculate FLOPs and Params
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.2f")
    with torch.no_grad():
        for _ in range(9):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        test_acc = inference(model=model, image_processor=image_processor,
                             data_loader=data_loader, device=device)
        
    # 确保 gpu 和 cpu 同步运行，因为计时由 cpu 负责，避免 cpu 无任务直接返回
    torch.cuda.synchronize()
        
    # 计算性能指标
    inference_time = time.time() - start_time
    throughput = total_images / inference_time if inference_time > 0 else 0
    
    # 打印结果
    print(f"\n=== Performance Metrics ===")
    print(f"Device: {device}")
    print(f"FLOPs: {flops}, Params: {params}")
    print(f"Total images processed: {total_images}")
    print(f"Accuracy: {test_acc*100:.2f}%")
    print(f"Total inference time: {inference_time:.4f} seconds")
    print(f"Throughput: {throughput:.2f} im/s")
    
    # 保存结果到文件
    if args.results_save_path:
        results = {
            "device": str(device),
            "flops": flops,
            "params": params,
            "total_images": total_images,
            "accuracy": test_acc,
            "total_inference_time": inference_time,
            "throughput": throughput  
        }
        
        with open(os.path.join(args.results_save_path, f"performance_tome_r-{args.tome_r}.json"), "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', 
                        type=str,
                        default='cuda:0',
                        help='Device to use for inference')
    parser.add_argument('--pretrained-model-path', 
                        type=str,
                        default='./checkpoints/vit-base-patch16-384',
                        help='Path to the pretrained model')
    parser.add_argument('--dataset-path', 
                        type=str,
                        default='./dataset/imagenet1k',
                        help='Path to the dataset')
    parser.add_argument('--results-save-path',
                        type=str,
                        default='./workdir/performance',
                        help='Save results to JSON file')
    parser.add_argument('--tome-r',
                        type=int,
                        default=0,
                        help='Token merging\'s hyper-parameter')
    args = parser.parse_args()

    for i in range(0, 17):
        args.tome_r = i
        print(f"------------------tome_r: {i}------------------")
        evaluate(args)