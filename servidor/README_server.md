# IA Image Server (CPU-only)

Este servidor expone una API REST que permite realizar detección de objetos y búsquedas semánticas sobre imágenes utilizando modelos de visión por computador como YOLOv8, Faster R-CNN, SSD y CLIP. Está pensado para funcionar en entornos forenses, como parte de un sistema cliente-servidor con Autopsy.

## Requisitos

- Docker (versión 20.10 o superior)
- Carpeta local `models/` con los pesos necesarios:
  - `models/yolov8n.pt`
  - `models/yolo11n.pt`
- Acceso a CPU suficiente para realizar inferencias con Torch y Transformers
- Sistema operativo compatible con Docker (Linux, Windows, WSL2, etc.)

## Construcción y despliegue

1. **Construcción de la imagen Docker**

```bash
docker build -t ai-server-cpu:latest .
```

2. **Ejecución del contenedor**

```bash
docker run -d ^
  --name ai-server-cpu ^
  -p 8000:8000 ^
  -v "%cd%\models:/app/models" ^
  ai-server-cpu:latest
```

> En Linux/Mac, cambia las barras invertidas por continuaciones con `\` y usa `$(pwd)` en lugar de `%cd%`.

3. **Verificar el arranque**

```bash
docker logs -f ai-server-cpu
```

4. **Gestión del contenedor**

```bash
docker stop ai-server-cpu
docker start ai-server-cpu
docker restart ai-server-cpu
```

5. **Eliminación del contenedor (si ya no se usa)**

```bash
docker stop ai-server-cpu
docker rm ai-server-cpu
```

## Endpoints disponibles

- `POST /process`: Recibe imágenes codificadas en base64 junto con el modelo a usar y devuelve los resultados en JSON.

## Modelos soportados

- `yv8` : YOLOv8 Nano
- `yv11`: YOLOv11 Nano (si se entrena o añade manualmente)
- `frcnn`: Faster R-CNN ResNet50 FPN
- `ssd`: SSD300 con VGG16
- `clip_img`: Comparación visual basada en CLIP
- `clip_txt`: Búsqueda por texto natural basada en CLIP