import ast
import glob
import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import average_precision_score, roc_curve
from torchvision.transforms import Compose, Resize, ToTensor


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
EOMT_ROOT = PROJECT_ROOT / "eomt"
sys.path.insert(0, str(EOMT_ROOT))


DATASETS = {
    "RoadAnomaly": "data/Validation_Dataset/RoadAnomaly/images/*",
    "RoadAnomaly21": "data/Validation_Dataset/RoadAnomaly21/images/*",
    "fs_static": "data/Validation_Dataset/fs_static/images/*",
    "LostFound": "data/Validation_Dataset/FS_LostFound_full/images/*",
    "RoadObsticle21": "data/Validation_Dataset/RoadObsticle21/images/*",
}


def fpr_at_95_tpr(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    if np.all(tpr < 0.95):
        return 1.0
    return float(fpr[np.argmax(tpr >= 0.95)])


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mapping(mapping_file):
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = ast.literal_eval(f.read())
    return {int(coco_id): int(city_id) for coco_id, city_id in mapping.items()}


def download_checkpoint(config):
    from huggingface_hub import hf_hub_download

    name = config["trainer"]["logger"]["init_args"]["name"]
    if name == "coco_panoptic_eomt_base_640":
        name = "coco_panoptic_eomt_base_640_2x"
    return hf_hub_download(
        repo_id=f"tue-mps/{name}",
        filename="pytorch_model.bin",
    )


def build_eomt_model(config, checkpoint_path, img_size, num_classes, device):
    import importlib

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {
        k: v for k, v in network_cfg["init_args"].items() if k != "encoder"
    }
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {
        k: v for k, v in config["model"]["init_args"].items() if k != "network"
    }
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = lit_cls(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    ).eval().to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded EoMT COCO checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return model


def gt_path_from_image_path(path):
    path_gt = path.replace("images", "labels_masks")
    if "RoadObsticle21" in path_gt:
        path_gt = path_gt.replace("webp", "png")
    if "fs_static" in path_gt:
        path_gt = path_gt.replace("jpg", "png")
    if "RoadAnomaly" in path_gt:
        path_gt = path_gt.replace("jpg", "png")
    return path_gt


def load_anomaly_gt(path_gt, img_size):
    mask = Image.open(path_gt)
    mask = Resize(img_size, interpolation=Image.NEAREST)(mask)
    ood_gts = np.array(mask)

    if "RoadAnomaly" in path_gt:
        ood_gts = np.where(ood_gts == 2, 1, ood_gts)
    if "LostAndFound" in path_gt:
        ood_gts = np.where(ood_gts == 0, 255, ood_gts)
        ood_gts = np.where(ood_gts == 1, 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    if "Streethazard" in path_gt:
        ood_gts = np.where(ood_gts == 14, 255, ood_gts)
        ood_gts = np.where(ood_gts < 20, 0, ood_gts)
        ood_gts = np.where(ood_gts == 255, 1, ood_gts)

    return ood_gts


def map_coco_to_cityscapes(pixel_probs, pixel_logits, mapping, num_city_classes):
    height, width = pixel_probs.shape[1], pixel_probs.shape[2]
    city_probs = torch.zeros(
        num_city_classes,
        height,
        width,
        device=pixel_probs.device,
        dtype=pixel_probs.dtype,
    )
    city_logits = torch.full(
        (num_city_classes, height, width),
        -1e9,
        device=pixel_logits.device,
        dtype=pixel_logits.dtype,
    )

    for coco_id, city_id in mapping.items():
        city_probs[city_id] += pixel_probs[coco_id]
        city_logits[city_id] = torch.maximum(city_logits[city_id], pixel_logits[coco_id])

    return city_probs, city_logits


def forward_eomt(model, image_path, img_size, device, mapping, num_city_classes):
    transform = Compose([Resize(img_size), ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device) * 255.0

    with torch.no_grad():
        mask_logits_per_layer, class_logits_per_layer = model(image_tensor)

    mask_logits = mask_logits_per_layer[-1]
    class_logits = class_logits_per_layer[-1]
    mask_logits = F.interpolate(
        mask_logits,
        size=img_size,
        mode="bilinear",
        align_corners=False,
    )

    mask_probs = mask_logits.sigmoid()
    class_probs = class_logits.softmax(dim=-1)[..., :-1]
    pixel_probs = torch.einsum("bqhw,bqc->bchw", mask_probs, class_probs)
    pixel_logits = torch.einsum("bqhw,bqc->bchw", mask_probs, class_logits[..., :-1])

    city_probs, city_logits = map_coco_to_cityscapes(
        pixel_probs[0],
        pixel_logits[0],
        mapping,
        num_city_classes,
    )
    return city_probs, city_logits


def anomaly_scores(pixel_probs, pixel_logits, methods):
    scores = {}
    if "msp" in methods:
        scores["msp"] = 1.0 - pixel_probs.max(dim=0).values
    if "entropy" in methods:
        prob_dist = pixel_probs / pixel_probs.sum(dim=0, keepdim=True).clamp_min(1e-8)
        entropy = -(prob_dist * prob_dist.clamp_min(1e-8).log()).sum(dim=0)
        scores["entropy"] = entropy / np.log(pixel_probs.shape[0])
    if "maxlogit" in methods:
        scores["maxlogit"] = -pixel_logits.max(dim=0).values
    if "rba" in methods:
        scores["rba"] = -pixel_probs.tanh().sum(dim=0)
    return {k: v.detach().cpu().numpy().astype("float32") for k, v in scores.items()}


def evaluate_dataset(model, dataset_path, methods, img_size, device, mapping, num_city_classes):
    score_values = {method: [] for method in methods}
    label_values = []
    paths = sorted(glob.glob(os.path.expanduser(dataset_path)))

    for path in paths:
        print(path)
        path_gt = gt_path_from_image_path(path)
        ood_gts = load_anomaly_gt(path_gt, img_size)
        if 1 not in np.unique(ood_gts):
            continue

        pixel_probs, pixel_logits = forward_eomt(
            model,
            path,
            img_size,
            device,
            mapping,
            num_city_classes,
        )
        scores = anomaly_scores(pixel_probs, pixel_logits, methods)

        valid_mask = (ood_gts == 0) | (ood_gts == 1)
        labels = (ood_gts[valid_mask] == 1).astype(np.uint8)
        label_values.append(labels)

        for method in methods:
            score_values[method].append(scores[method][valid_mask])

        del pixel_probs, pixel_logits, scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not label_values:
        raise RuntimeError(f"No valid anomaly labels found for dataset path: {dataset_path}")

    labels = np.concatenate(label_values)
    results = {}
    for method in methods:
        values = np.concatenate(score_values[method])
        auprc = average_precision_score(labels, values)
        fpr95 = fpr_at_95_tpr(values, labels)
        results[method] = {"auprc": auprc, "fpr95": fpr95}
    return results


def default_checkpoint_name(config, checkpoint_path):
    if checkpoint_path is not None:
        return Path(checkpoint_path).stem
    return config["trainer"]["logger"]["init_args"]["name"]


def main():
    parser = ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=["msp"], choices=["msp", "entropy", "maxlogit", "rba"])
    parser.add_argument(
        "--config",
        default=str(EOMT_ROOT / "configs" / "dinov2" / "coco" / "panoptic" / "eomt_base_640_2x.yaml"),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--mapping-file", default=str(PROJECT_ROOT / "notebooks" / "coco_to_cityscapes_mapping.json"))
    parser.add_argument("--img-size", type=int, nargs=2, default=(640, 640), metavar=("H", "W"))
    parser.add_argument("--num-classes", type=int, default=133)
    parser.add_argument("--num-city-classes", type=int, default=19)
    parser.add_argument("--results-file", default="results_eomt_coco_mapped.txt")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    img_size = tuple(args.img_size)

    config = load_config(args.config)
    mapping = load_mapping(args.mapping_file)
    checkpoint_path = args.checkpoint or download_checkpoint(config)
    checkpoint_name = args.checkpoint_name or default_checkpoint_name(config, args.checkpoint)
    model = build_eomt_model(config, checkpoint_path, img_size, args.num_classes, device)

    dataset_list = []
    if args.datasets is not None:
        for dataset_name in args.datasets:
            dataset_list.append((dataset_name, DATASETS[dataset_name]))
    else:
        dataset_list = [("RoadAnomaly21", DATASETS["RoadAnomaly21"])]

    with open(args.results_file, "a", encoding="utf-8") as file:
        file.write(f"\nEoMT COCO mapped anomaly evaluation | checkpoint={checkpoint_name}\n")
        file.write(f"mapping={args.mapping_file}\n")
        header = f"{'checkpoint':<18} {'dataset':<15} {'method':<8} {'AUPRC':>10} {'FPR@TPR95':>10}"
        print(header)
        file.write(header + "\n")
        for dataset_name, dataset_path in dataset_list:
            print("\nDataset:", dataset_name)
            results = evaluate_dataset(
                model,
                dataset_path,
                args.methods,
                img_size,
                device,
                mapping,
                args.num_city_classes,
            )
            for method, metrics in results.items():
                line = (
                    f"{checkpoint_name:<18} {dataset_name:<15} {method:<8} "
                    f"{metrics['auprc'] * 100.0:10.4f} "
                    f"{metrics['fpr95'] * 100.0:10.4f}"
                )
                print(line)
                file.write(line + "\n")


if __name__ == "__main__":
    main()
