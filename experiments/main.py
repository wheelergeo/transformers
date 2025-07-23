from src import vit_base_patch16_384_tome as vit_tome
from transformers import ViTForImageClassification, ViTImageProcessor, pipeline


import argparse


def pipeline_example():
    model = ViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path="./checkpoints/vit-base-patch16-384",
        use_safetensors=True,)

    image_processor = ViTImageProcessor.from_pretrained(
        pretrained_model_name_or_path="./checkpoints/vit-base-patch16-384",
    )

    pipeline = pipeline(task="visual-question-answering", model=model,
                        image_processor=image_processor, device_map="auto")

    output = pipeline("./dataset/imagenet1k/val/n01440764/ILSVRC2012_val_00000293.JPEG")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()