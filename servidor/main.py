#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import base64
import tempfile
import multiprocessing
import logging
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union

import psutil
from PIL import Image
import torchvision.transforms as T
import clip
from ultralytics import YOLO
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    ssd300_vgg16,
    SSD300_VGG16_Weights
)

from ia_detect import (
    detectar_yolo, detectar_frcnn_ssd,
    batch_detect_yolo, batch_detect_frcnn_ssd,
    batch_similitud_clip_img, batch_similitud_clip_txt
)

# — Silenciar logs ultralytics salvo errores —
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

# ----------------------------------------
# CONFIG GLOBAL: detectar recursos máquina
# ----------------------------------------
CPU_CORES  = multiprocessing.cpu_count()
TOTAL_MEM  = psutil.virtual_memory().total    # en bytes

# Modelos soportados
MODEL_FLAGS = ["clip_txt","yv11","frcnn","ssd","clip_img","yv8"]

# Batch sizes calculadas en startup
BATCH_SIZES = {}

SAFE_RAM_FRAC = 0.2   # reserva solo el 20% de la RAM total
MEM_FOOTPRINT = {}    # se rellenará en startup_event()

MIN_FOOTPRINT    = 1

# ----------------------------------------
# FASTAPI & Pydantic
# ----------------------------------------
app = FastAPI(title="IA Batch Server (CPU-only)")

class ImageItem(BaseModel):
    filename: str
    data:     str  # base64

class ReferenceItem(BaseModel):
    filename: str
    data:     str  # base64 or plain text (clip_txt)

class ProcessRequest(BaseModel):
    model:     str
    mode:      str  # "objects" | "clip_img" | "clip_txt"
    reference: Optional[Union[ReferenceItem,str]] = None
    images:    List[ImageItem]

class ObjectResult(BaseModel):
    class_name: str
    confidence: float

class ImageResult(BaseModel):
    filename:   str
    objects:    Optional[List[ObjectResult]] = None
    similarity: Optional[float] = None

class ProcessResponse(BaseModel):
    results: List[ImageResult]

# ----------------------------------------
# CARGA DE MODELOS
# ----------------------------------------
_models = {}
def cargar_modelo(flag: str):
    f = flag.lower()
    if f in ("yv8","yv11"):
        prefix = {"yv8":"yolov8n","yv11":"yolo11n"}[f]
        pt = glob.glob(f"/app/models/{prefix}.pt")[0]
        _models[f] = YOLO(pt, verbose=False)
    elif f == "frcnn":
        w = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        m = fasterrcnn_resnet50_fpn(weights=w); m.eval()
        _models[f] = (m, w.meta["categories"])
    elif f == "ssd":
        w = SSD300_VGG16_Weights.DEFAULT
        m = ssd300_vgg16(weights=w); m.eval()
        _models[f] = (m, w.meta["categories"])
    elif f in ("clip_img","clip_txt"):
        device = "cpu"
        m,pre = clip.load("ViT-B/32", device=device)
        _models[f] = (m, pre, device)
    else:
        raise ValueError(f"Modelo no válido: {flag}")
    return _models[f]

def get_model(flag: str):
    return _models.get(flag.lower()) or cargar_modelo(flag)

# ----------------------------------------
# ARRANQUE: precarga + autotune CPU
# ----------------------------------------
   
@app.on_event("startup")
def startup_event():
     avail_ram = psutil.virtual_memory().available
     safe_ram  = avail_ram * SAFE_RAM_FRAC
     cpu_dev   = torch.device("cpu")

     for flag in MODEL_FLAGS:
         entry     = get_model(flag)
         proc      = psutil.Process()
         footprint = 0

         # seguimos midiendo hasta que sea un valor razonable
         while True:
             before = proc.memory_info().rss
             try:
                 if flag in ("yv8","yv11"):
                     dummy = torch.zeros((1,3,640,640), device=cpu_dev)
                     with torch.no_grad(): entry(dummy)

                 elif flag in ("frcnn","ssd"):
                     model, _ = entry
                     img = Image.new("RGB",(640,640))
                     t   = T.ToTensor()(img).unsqueeze(0)
                     with torch.no_grad(): model(t)

                 else:  # clip_img / clip_txt
                     m, pre, dev = entry
                     img   = Image.new("RGB",(224,224))
                     inp   = pre(img).unsqueeze(0)
                     with torch.no_grad():
                         if flag == "clip_img":
                             m.encode_image(inp)
                         else:
                             m.encode_text(clip.tokenize(["texto"]))

             except Exception:
                 # si falla la inferencia dummy, lo registramos y seguimos
                 logger.warning(f"Fallo en infer dummy para {flag}, reintentando...")
             after   = proc.memory_info().rss
             delta   = after - before

             # comprobamos si está dentro de límites sensatos
             if MIN_FOOTPRINT < delta <= safe_ram:
                 footprint = delta
                 logger.warning(
                     f"Medición valida para {flag}: {delta} bytes "
                 )
                 break
             else:
                 logger.warning(
                     f"Medición inválida para {flag}: {delta} bytes "
                     f"(esperado entre {MIN_FOOTPRINT} y {safe_ram}). Reintentando..."
                 )

         # guardamos footprint y calculamos batch
         MEM_FOOTPRINT[flag] = footprint
         max_by_ram = int(safe_ram // footprint)
         logger.warning(f"Medición max_by_ram para {flag}: {max_by_ram} y cpu_cores: {CPU_CORES}")
         BATCH_SIZES[flag]  = max(1, min(CPU_CORES, max_by_ram))
         #BATCH_SIZES[flag]  = max_by_ram/2 #Si no queremos que se limite a los cores de CPU
     print(" MEM_FOOTPRINT (bytes):", MEM_FOOTPRINT, flush=True)
     print(" BATCH_SIZES auto-tunados:", BATCH_SIZES, flush=True)

# ----------------------------------------
# UTILIDADES
# ----------------------------------------
def decode_all(images: List[ImageItem]) -> List[str]:
    paths = []
    for img in images:
        raw = base64.b64decode(img.data)
        suf = os.path.splitext(img.filename)[1]
        tmp = tempfile.NamedTemporaryFile(suffix=suf, delete=False)
        tmp.write(raw); tmp.flush(); tmp.close()
        paths.append(tmp.name)
    return paths

def cleanup(paths: List[str]):
    for p in paths:
        try: os.unlink(p)
        except: pass

def select_strategy(flag: str) -> str:
    return "batch" if BATCH_SIZES.get(flag.lower(),1) > 1 else "single"

def get_dynamic_bs(flag: str) -> int:
    avail = psutil.virtual_memory().available * SAFE_RAM_FRAC
    bs = int(avail // MEM_FOOTPRINT.get(flag, 1))
    result = max(1, min(CPU_CORES, bs))
    #result = int(bs/2) #Si no queremos que se limite a los cores de CPU
    logger.warning(
                    f"Nueva medición para {flag}: {bs} batch-bytes and CPU: {CPU_CORES}, nueva batch size: {result} "
                )
    return result
# ----------------------------------------
# ENDPOINT /process
# ----------------------------------------
@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    tmp_paths = decode_all(req.images)
    results   = [None] * len(tmp_paths)
    ref_path  = None

    if req.mode == "clip_img":
        item = ReferenceItem(**req.reference.dict()) if hasattr(req.reference, "dict") else req.reference
        raw  = base64.b64decode(item.data)
        suf  = os.path.splitext(item.filename)[1]
        tf   = tempfile.NamedTemporaryFile(suffix=suf, delete=False)
        tf.write(raw); tf.flush(); tf.close()
        ref_path = tf.name

    try:
        if req.mode == "objects":
            flag = req.model.lower()
            fn   = batch_detect_yolo if flag in ("yv8","yv11") else batch_detect_frcnn_ssd

            # calculamos dinámicamente el batch por RAM libre
            bs = int(get_dynamic_bs(flag))

            # intentamos en modo batch, reduciendo si OOM
            while bs > 1:
                try:
                    chunks = [ tmp_paths[i:i+bs] for i in range(0, len(tmp_paths), bs) ]
                    for ci,chunk in enumerate(chunks):
                        outs = fn(get_model(flag), chunk)
                        for j,det in enumerate(outs):
                            idx = ci*bs + j
                            results[idx] = ImageResult(
                                filename=req.images[idx].filename,
                                objects=[ObjectResult(class_name=k, confidence=v)
                                         for k,v in det.items()]
                            )
                    break
                except RuntimeError:
                    logger.warning(f"OOM en batch {flag} bs={bs} → retry bs={bs//2}")
                    bs //= 2

            # si no cupo ni siquiera bs=2, caer a single
            if bs <= 1:
                for i,path in enumerate(tmp_paths):
                    det = (detectar_yolo(get_model(flag), path)
                           if flag in ("yv8","yv11")
                           else detectar_frcnn_ssd(get_model(flag), path))
                    results[i] = ImageResult(
                        filename=req.images[i].filename,
                        objects=[ObjectResult(class_name=k, confidence=v)
                                 for k,v in det.items()]
                    )

        elif req.mode == "clip_img":
            flag = "clip_img"; fn = batch_similitud_clip_img
            bs   = get_dynamic_bs(flag)
            while bs > 1:
                try:
                    chunks = [ tmp_paths[i:i+bs] for i in range(0, len(tmp_paths), bs) ]
                    for ci,chunk in enumerate(chunks):
                        sims = fn(get_model(flag), ref_path, chunk)
                        for j,s in enumerate(sims):
                            idx = ci*bs + j
                            results[idx] = ImageResult(
                                filename=req.images[idx].filename,
                                similarity=s
                            )
                    break
                except RuntimeError:
                    logger.warning(f"OOM en clip_img bs={bs} → retry bs={bs//2}")
                    bs //= 2
            if bs <= 1:
                for i,path in enumerate(tmp_paths):
                    s = fn(get_model(flag), ref_path, [path])[0]
                    results[i] = ImageResult(filename=req.images[i].filename, similarity=s)

        elif req.mode == "clip_txt":
            flag = "clip_txt"; fn = batch_similitud_clip_txt
            bs   = get_dynamic_bs(flag)
            while bs > 1:
                try:
                    chunks = [ tmp_paths[i:i+bs] for i in range(0, len(tmp_paths), bs) ]
                    for ci,chunk in enumerate(chunks):
                        texts = [req.reference] * len(chunk)
                        sims = fn(get_model(flag), texts, chunk)
                        for j,s in enumerate(sims):
                            idx = ci*bs + j
                            results[idx] = ImageResult(
                                filename=req.images[idx].filename,
                                similarity=s
                            )
                    break
                except RuntimeError:
                    logger.warning(f"OOM en clip_txt bs={bs} → retry bs={bs//2}")
                    bs //= 2
            if bs <= 1:
                for i,path in enumerate(tmp_paths):
                    s = fn(get_model(flag), [req.reference], [path])[0]
                    results[i] = ImageResult(filename=req.images[i].filename, similarity=s)

        else:
            raise HTTPException(400, f"Unknown mode: {req.mode}")

        return ProcessResponse(results=results)

    except Exception:
        logger.exception("Error en /process")
        raise HTTPException(500, "Error interno")

    finally:
        cleanup(tmp_paths)
        if ref_path:
            try: os.unlink(ref_path)
            except: pass
