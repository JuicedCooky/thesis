import torch, os 
import csv
from . import evaluation
import clip.clip as clip 

def test(args):
    checkpoint = torch.load(args.load, map_location=torch.device("cpu"))
    model_iteration_count = checkpoint["iteration"]
    print(model_iteration_count)
    print(args.save + "/" + f"metrics_{model_iteration_count}.csv")


    model,val_preprocess,_ = clip.load("ViT-B/16", jit=False)
    
    results = evaluation.evaluate_2(model, args, val_preprocess)

    with open(args.save + "/" + f"metrics_{model_iteration_count}.csv", mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration","dataset_name","top1","top5"])
        writer.writeheader()
        writer.writerows(results)
