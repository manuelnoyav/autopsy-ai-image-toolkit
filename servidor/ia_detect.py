#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T
import clip
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    ssd300_vgg16, SSD300_VGG16_Weights
)

import torchvision.transforms.functional as F

# ------------------------------------------------
# Model loading & single-image inference (CPU-only)
# ------------------------------------------------
_models = {}

BatchList = {}

def cargar_modelo(model_flag: str):
    flag = model_flag.lower()

    if flag in ("yv8", "yv11"):
        prefijos = {"yv8": "yolov8n", "yv11": "yolo11n"}
        prefix = prefijos[flag]
        pattern = f"./models/{prefix}.pt" # app/models/{prefix}.pt desde la vision de docker
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f"No se encuentra ningún peso YOLO para '{flag}' en {pattern}")
        pt = candidates[0]
        model = YOLO(pt)
        model.to('cpu')
        _models[flag] = model
        BatchList[flag] = 4
        return model

    elif flag == "frcnn":
        w = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=w)
        model.to('cpu')
        model.eval()
        _models[flag] = (model, w.meta["categories"])
        BatchList[flag] = 2
        return _models[flag]

    elif flag == "ssd":
        w = SSD300_VGG16_Weights.DEFAULT
        model = ssd300_vgg16(weights=w)
        model.to('cpu')
        model.eval()
        _models[flag] = (model, w.meta["categories"])
        BatchList[flag] = 4
        return _models[flag]

    elif flag in ("clip_img", "clip_txt"):
        device = 'cpu'
        m, preprocess = clip.load("ViT-B/32", device=device)
        _models[flag] = (m, preprocess, device)
        BatchList[flag] = 8 if flag == "clip_img" else 16
        return _models[flag]

    else:
        raise ValueError(f"Modelo no válido: {model_flag}")

def get_model(model_flag: str):
    return _models.get(model_flag) or cargar_modelo(model_flag)

def detectar_yolo(model: YOLO, image_path: str) -> dict:
    results = model.predict(source=image_path, verbose=False, show=False)
    det = {}
    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            name = model.names[cls]
            det.setdefault(name, []).append(conf)
    if not det:
        return {"ImposibleClasificar": 1.0}
    return {k: round(sum(v)/len(v), 4) for k, v in det.items()}

def detectar_frcnn_ssd(model_data, image_path: str) -> dict:
    model, labels = model_data
    img = Image.open(image_path).convert("RGB")
    tensor = T.ToTensor()(img).unsqueeze(0).to('cpu')
    with torch.no_grad():
        out = model(tensor)[0]
    if 'boxes' not in out or out['boxes'].nelement() == 0:
        return {"ImposibleClasificar": 1.0}
    det = {}
    for box, lbl, score in zip(out['boxes'], out['labels'], out['scores']):
        name = labels[lbl.item()] if lbl.item() < len(labels) else f"cls_{lbl.item()}"
        det.setdefault(name, []).append(score.item())

    # Si tras acumular no hay etiquetas, marco imposible
    if not det:
        return {"ImposibleClasificar": 1.0}

    # Saco el promedio de confianza por etiqueta
    return {k: round(sum(v) / len(v), 4) for k, v in det.items()}

def similitud_clip_img(model_data, ref_path: str, img_path: str) -> float:
    model, pre, device = model_data
    a = pre(Image.open(ref_path)).unsqueeze(0).to(device)
    b = pre(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        e1 = model.encode_image(a)
        e2 = model.encode_image(b)
    return round(torch.cosine_similarity(e1, e2).item(), 4)

def similitud_clip_txt(model_data, text: str, img_path: str) -> float:
    model, pre, device = model_data
    img = pre(Image.open(img_path)).unsqueeze(0).to(device)
    txt = clip.tokenize([text]).to(device)
    with torch.no_grad():
        e_img = model.encode_image(img)
        e_txt = model.encode_text(txt)
    return round(torch.cosine_similarity(e_txt, e_img).item(), 4)

def batch_detect_yolo(model: YOLO, image_paths: list) -> list:
    results = model.predict(source=image_paths, batch=BatchList.get("yv8", 1), verbose=False, show=False)
    salida = []
    for r in results:
        det = {}
        for b in r.boxes:
            cls = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            name = model.names[cls]
            det.setdefault(name, []).append(conf)
        salida.append({k: round(sum(v)/len(v), 4) for k, v in det.items()} or {"ImposibleClasificar": 1.0})
    return salida

def batch_detect_frcnn_ssd(model_data, image_paths: list) -> list:
    model, labels = model_data
    device = next(model.parameters()).device

    # 1) Lee todas las imágenes y obtiene sus tamaños originales
    sizes = [Image.open(p).size for p in image_paths]  # (W, H)
    max_w = max(w for w, h in sizes)
    max_h = max(h for w, h in sizes)

    # 2) Construye un tensor por imagen, paddeado a (max_h, max_w)
    tensors = []
    for path, (w, h) in zip(image_paths, sizes):
        img = Image.open(path).convert("RGB")
        t = T.ToTensor()(img)  # [3, H, W]
        # pad = (left, top, right, bottom)
        pad = (0, 0, max_w - w, max_h - h)
        t = F.pad(t, pad, fill=0)  # ahora [3, max_h, max_w], con bordes negros
        tensors.append(t.to(device))

    # 3) Forward en batch
    batch = torch.stack(tensors, dim=0)
    with torch.no_grad():
        outputs = model(batch)

    # 4) Procesa cada salida
    results = []
    for out in outputs:
        # Si no hay absolutamente ninguna caja
        if 'boxes' not in out or out['boxes'].nelement() == 0:
            results.append({"ImposibleClasificar": 1.0})
            continue
        # Acumular todas las detecciones sin filtrar
        det = {}
        for box, lbl, score in zip(out['boxes'], out['labels'], out['scores']):
            name = labels[lbl.item()] if lbl.item() < len(labels) else f"cls_{lbl.item()}"
            det.setdefault(name, []).append(score.item())
        # Si tras acumular no hay etiquetas, marco imposible
        if not det:
            results.append({"ImposibleClasificar": 1.0})
        else:
            results.append({k: round(sum(v) / len(v), 4) for k, v in det.items()})

    return results

def batch_similitud_clip_img(model_data, ref_path: str, image_paths: list) -> list:
    model, pre, device = model_data
    ref = pre(Image.open(ref_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        ref_emb = model.encode_image(ref)
    imgs = [pre(Image.open(p)).unsqueeze(0) for p in image_paths]
    batch = torch.cat(imgs, 0).to(device)
    with torch.no_grad():
        img_embs = model.encode_image(batch)
    sims = torch.cosine_similarity(ref_emb, img_embs, dim=1)
    return [round(float(s), 4) for s in sims]

def batch_similitud_clip_txt(model_data, texts: list, image_paths: list) -> list:
    model, pre, device = model_data
    imgs = [pre(Image.open(p)).unsqueeze(0) for p in image_paths]
    batch = torch.cat(imgs, 0).to(device)
    txts = clip.tokenize(texts).to(device)
    with torch.no_grad():
        img_embs = model.encode_image(batch)
        txt_embs = model.encode_text(txts)
    sims = torch.cosine_similarity(txt_embs, img_embs, dim=1)
    return [round(float(s), 4) for s in sims]