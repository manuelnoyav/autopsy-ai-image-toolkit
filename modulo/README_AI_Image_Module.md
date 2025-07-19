#  Autopsy AI Image Module

Este módulo permite integrar capacidades de visión por computador en Autopsy, enviando imágenes extraídas del caso a un servidor externo que devuelve resultados procesados por modelos de IA.

---

## Requisitos del sistema

Para ejecutar correctamente este módulo, es necesario contar con:

- **Sistema operativo**: Windows 10 (probado en versión 22H2).
- **Autopsy**: Versión 4.19 (con soporte para Python plugins vía Jython).
- **Jython**: Versión 2.7.4  
  - [Descargar desde https://www.jython.org/download](https://www.jython.org/download)
- **Java**: OpenJDK 17 (probado con Temurin-17.0.14)  
- **Python**: Aunque el módulo se ejecuta en Jython, puedes necesitar Python 3.11.5.

---

## Instalación

1. **Clonar o descargar el módulo** desde el repositorio GitHub.

2. **Copiar la carpeta** del módulo (`AI_Image_Module/`) al directorio de plugins de Autopsy para Python:

   ```bash
   C:\Users\<tu_usuario>\AppData\Roaming\Autopsy\python_modules\
   ```

3. **Configurar la URL del servidor de IA y el archivo de LOG**:  
   Abre el archivo `AI_Image_Module.py` y edita la línea **374** para apuntar al endpoint correcto de tu servidor:

   ```python
   self.server_url = "http://localhost:8000/process"  # Línea 376
   ```

   Si tu servidor está en otra IP o puerto, cámbialo aquí.

   Para el archivo de logs especifica la ruta:
	
   ```python
   LOG_FILE_PATH = r"C:\Users\<tu_usuario>\Desktop\logs.txt" # Línea 39
   ```
4. **Iniciar Autopsy**, cargar un caso y verificar que el módulo aparece en el panel de ingest modules.

---

## Dependencias del módulo

El código está escrito en Python 2.7 (compatible con Jython), por lo que **no necesita instalación adicional de paquetes vía pip**. Sin embargo, usa las siguientes librerías estándar:

- `json`
- `urllib2`
- `base64`
- `java.io`, `java.lang`, `java.awt.image`, `javax.imageio`, etc. (para integración con Java/Autopsy)
- `org.sleuthkit.datamodel` y `org.sleuthkit.autopsy.ingest` (clases internas de Autopsy)

>  Todas estas dependencias están disponibles por defecto en el entorno Jython + Autopsy.

---