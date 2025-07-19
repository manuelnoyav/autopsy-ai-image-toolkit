# -*- coding: utf-8 -*-
###############################################################################
# IMPORTS
###############################################################################
# Importaciones de Java para manejo de arrays, inspeccion y utilidades del sistema
import jarray
import inspect
import os
import json
import codecs
import shutil
import io

# Importaciones de Java para manejo de archivos y flujos de datos
from java.io import FileOutputStream, OutputStreamWriter, File as JavaFile
from java.util.logging import Level

# Importaciones de Autopsy para configuracion y manejo de modulos
from org.sleuthkit.autopsy.ingest import GenericIngestModuleJobSettings, DataSourceIngestModuleAdapter
from org.sleuthkit.datamodel import BlackboardArtifact, BlackboardAttribute, AbstractFile, ReadContentInputStream
from org.sleuthkit.autopsy.ingest import IngestModule, IngestModuleFactoryAdapter, IngestModuleIngestJobSettingsPanel
from org.sleuthkit.autopsy.casemodule import Case
from org.sleuthkit.autopsy.coreutils import Logger

# Importaciones de Java Swing para la interfaz grafica
from javax.swing import JLabel, JPanel, ButtonGroup, JRadioButton, JButton, JFileChooser, BorderFactory, BoxLayout, JTextField
from java.awt import GridLayout, FlowLayout, Dimension

# Importaciones para realizar peticiones HTTP y codificacion Base64
from java.net import URL
from java.nio.file import Paths, Files
from java.util import Base64, ArrayList



###############################################################################
# LOGS
###############################################################################
LOG_FILE_PATH = r"C:\Users\<tu_usuario>\Desktop\logs.txt"

def file_log(message):
    """Escribe el mensaje en el fichero de log."""
    with codecs.open(LOG_FILE_PATH, "a", "utf-8") as f:
        f.write(message + "\n")


###############################################################################
# PANEL DE CONFIGURACIoN
###############################################################################
class AIModelSettingsPanel(IngestModuleIngestJobSettingsPanel):
    def __init__(self, local_settings):
        file_log("DEBUG PANEL: __init__() llamado.")
        super(AIModelSettingsPanel, self).__init__()
        if not hasattr(local_settings, "getSetting"):
            self.local_settings = GenericIngestModuleJobSettings()
            self.local_settings.setSetting("functionality", "objects")
            self.local_settings.setSetting("ai_model_flag", "Yv11")
            self.local_settings.setSetting("clip_image_path", "")
            self.local_settings.setSetting("clip_text_query", "")
        else:
            self.local_settings = local_settings
        self.initComponents()
        self.customizeComponents()

    def initComponents(self):
        file_log("DEBUG PANEL: initComponents() llamado.")
        self.setLayout(BoxLayout(self, BoxLayout.Y_AXIS))

        # PANEL 1: Funcionalidad principal
        func_panel = JPanel(FlowLayout(FlowLayout.LEFT, 3, 3))
        func_panel.setBorder(BorderFactory.createTitledBorder("Funcionalidad Principal"))
        self.obj_detection_rb = JRadioButton("Deteccion de Objetos", actionPerformed=self.selectMainFunctionality)
        self.similarity_rb = JRadioButton("Busqueda por Similitud", actionPerformed=self.selectMainFunctionality)
        self.func_group = ButtonGroup()
        self.func_group.add(self.obj_detection_rb)
        self.func_group.add(self.similarity_rb)
        func_subpanel = JPanel()
        func_subpanel.setLayout(BoxLayout(func_subpanel, BoxLayout.Y_AXIS))
        func_subpanel.add(self.obj_detection_rb)
        func_subpanel.add(self.similarity_rb)
        func_panel.add(func_subpanel)
        self.add(func_panel)

        # PANEL 2: Tipo de Similitud
        self.sim_type_panel = JPanel()
        self.sim_type_panel.setLayout(BoxLayout(self.sim_type_panel, BoxLayout.Y_AXIS))
        self.sim_type_panel.setBorder(BorderFactory.createTitledBorder("Tipo de Similitud"))

        self.clip_img_rb = JRadioButton("Por Imagen", actionPerformed=self.selectSimilarityType)
        self.clip_txt_rb = JRadioButton("Por Texto", actionPerformed=self.selectSimilarityType)

        self.sim_type_group = ButtonGroup()
        self.sim_type_group.add(self.clip_img_rb)
        self.sim_type_group.add(self.clip_txt_rb)

        self.sim_type_panel.add(self.clip_img_rb)
        self.sim_type_panel.add(self.clip_txt_rb)

        self.sim_type_panel.setPreferredSize(Dimension(200, self.sim_type_panel.getPreferredSize().height))

        # Contenedor con FlowLayout para buena alineacion
        sim_type_container = JPanel(FlowLayout(FlowLayout.LEFT, 3, 3))
        sim_type_container.add(self.sim_type_panel)

        self.sim_type_panel.setVisible(False)
        self.add(sim_type_container)

        # PANEL 3: Modelos de deteccion
        self.obj_models_panel = JPanel(GridLayout(2, 2, 10, 10))
        self.obj_models_panel.setBorder(BorderFactory.createTitledBorder("Modelos de Deteccion"))
        self.model_yv8_rb = JRadioButton("YOLOv8", actionPerformed=self.setAIModelFlag)
        self.model_yv11_rb = JRadioButton("YOLOv11", actionPerformed=self.setAIModelFlag)
        self.model_frcnn_rb = JRadioButton("Faster R-CNN", actionPerformed=self.setAIModelFlag)
        self.model_ssd_rb = JRadioButton("SSD", actionPerformed=self.setAIModelFlag)
        self.model_group = ButtonGroup()
        for rb in [self.model_yv8_rb, self.model_yv11_rb, self.model_frcnn_rb, self.model_ssd_rb]:
            self.model_group.add(rb)
            self.obj_models_panel.add(rb)
        self.add(self.obj_models_panel)

        # PANEL 3b: Umbral de confianza para Detección de Objetos
        self.obj_thresh_panel = JPanel(FlowLayout(FlowLayout.LEFT, 3, 3))
        self.obj_thresh_panel.setBorder(
            BorderFactory.createTitledBorder("Umbral Confianza")
        )
        self.obj_thresh_field = JTextField(5)
        self.obj_thresh_panel.add(JLabel("Confianza de Deteccion [0-100]%"))
        self.obj_thresh_panel.add(self.obj_thresh_field)
        self.obj_thresh_panel.setVisible(False)   # oculto por defecto
        self.add(self.obj_thresh_panel)

        # PANEL 4: Parametros de Similitud CLIP
        self.clip_panel = JPanel()
        self.clip_panel.setLayout(BoxLayout(self.clip_panel, BoxLayout.Y_AXIS))
        self.clip_panel.setBorder(BorderFactory.createTitledBorder("Referencia CLIP"))

        clip_img_text_subpanel = JPanel(FlowLayout(FlowLayout.LEFT, 3, 3))
        self.clip_img_btn = JButton("Elegir Imagen", actionPerformed=self.chooseClipImage)
        self.clip_img_label = JLabel("Imagen NO seleccionada")
        self.clip_txt_label = JLabel("Texto:")
        self.clip_txt_field = JTextField(15)
        clip_img_text_subpanel.add(self.clip_img_label)
        clip_img_text_subpanel.add(self.clip_img_btn)
        clip_img_text_subpanel.add(self.clip_txt_label)
        clip_img_text_subpanel.add(self.clip_txt_field)

        clip_threshold_subpanel = JPanel(FlowLayout(FlowLayout.LEFT, 3, 3))
        self.clip_threshold_label = JLabel("Confianza de Similitud [1 - 100]%")
        self.clip_threshold_field = JTextField(5)
        clip_threshold_subpanel.add(self.clip_threshold_label)
        clip_threshold_subpanel.add(self.clip_threshold_field)

        self.clip_panel.add(clip_img_text_subpanel)
        self.clip_panel.add(clip_threshold_subpanel)
        self.add(self.clip_panel)

        file_log("DEBUG PANEL: Componentes creados.")

    def selectMainFunctionality(self, event):
        if self.obj_detection_rb.isSelected():
            self.local_settings.setSetting("functionality", "objects")
            self.obj_models_panel.setVisible(True)
            self.sim_type_panel.setVisible(False)
            self.obj_thresh_panel.setVisible(True)
            self.obj_thresh_field.setText("")
            self.obj_thresh_field.setToolTipText("Valor por defecto: 0%")
            self.clip_panel.setVisible(False)
            self.sim_type_group.clearSelection()
            self.clip_threshold_field.setText("")
        elif self.similarity_rb.isSelected():
            self.obj_thresh_panel.setVisible(False)
            self.local_settings.setSetting("functionality", "similarity")
            self.obj_models_panel.setVisible(False)
            self.sim_type_panel.setVisible(True)
            self.clip_panel.setVisible(False)
        file_log("DEBUG PANEL: Main functionality seleccionada: {}".format(self.local_settings.getSetting("functionality")))

    def selectSimilarityType(self, event):
        if self.clip_img_rb.isSelected():
            self.local_settings.setSetting("functionality", "clip_img")
            self.local_settings.setSetting("ai_model_flag", "CLIP_IMG")
            self.local_settings.setSetting("clip_image_path", "")
            self.clip_img_label.setText("Imagen NO seleccionada")
            self.clip_img_btn.setVisible(True)
            self.clip_img_label.setVisible(True)
            self.clip_txt_label.setVisible(False)
            self.clip_txt_field.setVisible(False)
            self.clip_threshold_field.setToolTipText("Valor por defecto: 80%")
            self.clip_threshold_field.setText("")
            self.clip_panel.setVisible(True)
        elif self.clip_txt_rb.isSelected():
            self.local_settings.setSetting("functionality", "clip_txt")
            self.local_settings.setSetting("ai_model_flag", "CLIP_TXT")
            self.clip_txt_label.setVisible(True)
            self.clip_txt_field.setVisible(True)
            self.clip_img_btn.setVisible(False)
            self.clip_img_label.setVisible(False)
            self.clip_txt_field.setText("")
            self.clip_threshold_field.setToolTipText("Valor por defecto: 20%")
            self.clip_threshold_field.setText("")
            self.clip_panel.setVisible(True)

        file_log("DEBUG PANEL: Tipo de similitud seleccionado: functionality={}, ai_model_flag={}".format(
            self.local_settings.getSetting("functionality"), self.local_settings.getSetting("ai_model_flag")))

    def chooseClipImage(self, event):
        chooser = JFileChooser()
        result = chooser.showOpenDialog(self)
        if result == JFileChooser.APPROVE_OPTION:
            selected_path = chooser.getSelectedFile().getAbsolutePath()
            self.local_settings.setSetting("clip_image_path", selected_path)
            self.clip_img_label.setText("Imagen SI seleccionada")
            file_log("DEBUG PANEL: Imagen CLIP seleccionada: {}".format(selected_path))

    def customizeComponents(self):
        file_log("DEBUG PANEL: customizeComponents() llamado.")
        functionality = self.local_settings.getSetting('functionality')
        ai_model_flag = self.local_settings.getSetting('ai_model_flag')
        clip_path = self.local_settings.getSetting('clip_image_path')
        clip_text = self.local_settings.getSetting('clip_text_query')
        clip_threshold = self.local_settings.getSetting('clip_similarity_threshold')

        if clip_threshold:
            self.clip_threshold_field.setText(clip_threshold)
        else:
            self.clip_threshold_field.setText("")

        if functionality == "objects" or functionality is None:
            self.obj_detection_rb.setSelected(True)
            self.obj_models_panel.setVisible(True)
            self.obj_thresh_panel.setVisible(True)
            self.sim_type_panel.setVisible(False)
            self.clip_panel.setVisible(False)
        elif functionality == "similarity":
            self.similarity_rb.setSelected(True)
            self.obj_models_panel.setVisible(False)
            self.sim_type_panel.setVisible(True)
            self.obj_thresh_panel.setVisible(False)
            if ai_model_flag == "CLIP_IMG":
                self.clip_img_rb.setSelected(True)
                self.selectSimilarityType(None)
            elif ai_model_flag == "CLIP_TXT":
                self.clip_txt_rb.setSelected(True)
                self.selectSimilarityType(None)

        if ai_model_flag == "Yv8":
            self.model_yv8_rb.setSelected(True)
        elif ai_model_flag == "FRCNN":
            self.model_frcnn_rb.setSelected(True)
        elif ai_model_flag == "SSD":
            self.model_ssd_rb.setSelected(True)
        elif ai_model_flag == "Yv11":
            self.model_yv11_rb.setSelected(True)

        file_log("DEBUG PANEL: customizeComponents() finalizado.")

    def setAIModelFlag(self, event):
        if self.model_yv8_rb.isSelected():
            self.local_settings.setSetting('ai_model_flag', "Yv8")
        elif self.model_frcnn_rb.isSelected():
            self.local_settings.setSetting('ai_model_flag', "FRCNN")
        elif self.model_ssd_rb.isSelected():
            self.local_settings.setSetting('ai_model_flag', "SSD")
        else:
            self.local_settings.setSetting('ai_model_flag', "Yv11")
        file_log("DEBUG PANEL: Modelo seleccionado: {}".format(self.local_settings.getSetting('ai_model_flag')))

    def getSettings(self):
        file_log("DEBUG PANEL: getSettings() llamado.")
        self.local_settings.setSetting("objects_confidence_threshold", self.obj_thresh_field.getText())
        self.local_settings.setSetting("clip_text_query", self.clip_txt_field.getText())
        self.local_settings.setSetting("clip_similarity_threshold", self.clip_threshold_field.getText())
        return self.local_settings


###############################################################################
# FACTORIA DEL MODULO
###############################################################################
class AIObjectDetectionFactory(IngestModuleFactoryAdapter):
    moduleName = "AI Image Search & Detection"

    def getModuleDisplayName(self):
        return self.moduleName

    def getModuleDescription(self):
        return "Modulo IA para deteccion y busqueda de objetos."

    def getModuleVersionNumber(self):
        return "1.0"

    def isFileIngestModuleFactory(self):
        file_log("DEBUG Factory: isFileIngestModuleFactory() => False")
        return False

    def isDataSourceIngestModuleFactory(self):
        # DataSourceIngestModule
        file_log("DEBUG Factory: isDataSourceIngestModuleFactory() => True")
        return True

    def hasIngestJobSettingsPanel(self):
        file_log("DEBUG Factory: hasIngestJobSettingsPanel() => True")
        return True

    def getDefaultIngestJobSettings(self):
        file_log("DEBUG FACTORY: getDefaultIngestJobSettings() llamado. Creando GenericIngestModuleJobSettings con valores por defecto.")
        settings = GenericIngestModuleJobSettings()
        settings.setSetting("functionality", "objects")
        settings.setSetting("ai_model_flag", "Yv8")
        settings.setSetting("clip_image_path", "")
        settings.setSetting("clip_text_query", "")
        return settings

    def getIngestJobSettingsPanel(self, settings):
        file_log("DEBUG FACTORY: getIngestJobSettingsPanel() llamado. Ignorando ajustes anteriores y usando valores por defecto.")
        settings = self.getDefaultIngestJobSettings()
        return AIModelSettingsPanel(settings)

    def createDataSourceIngestModule(self, settings):
        file_log("DEBUG FACTORY: createDataSourceIngestModule() llamado.")
        current_flag = settings.getSetting("ai_model_flag")
        file_log("DEBUG FACTORY: createDataSourceIngestModule con ai_model_flag = {}".format(current_flag))
        return AIObjectDetectionModule(settings)

###############################################################################
# MODULO DE PROCESAMIENTO DE ARCHIVOS
###############################################################################
class AIObjectDetectionModule(DataSourceIngestModuleAdapter):
    _logger = Logger.getLogger("AI Object Detection")

    def __init__(self, settings):
        super(AIObjectDetectionModule, self).__init__()
        self.local_settings = settings
        self.imageItems = []
        self.filesProcessed = 0
        # configure logger method
        self.log = lambda level, msg: self._logger.logp(level, self.__class__.__name__, inspect.stack()[1][3], msg)
        flag = settings.getSetting("ai_model_flag") if hasattr(settings, "getSetting") else "N/A"
        file_log("DEBUG MODULE: __init__() called. ai_model_flag={}".format(flag))

    def startUp(self, context):
        # Guardamos el context para luego poder comprobar cancelacion
        self.context = context

        file_log("DEBUG MODULE: startUp() called.")
        self.model_flag = self.local_settings.getSetting("ai_model_flag") or "Yv11"
        self.functionality = self.local_settings.getSetting("functionality") or "objects"

        # Asignación de umbral según modalidad:
        if self.functionality == "objects":
            # Mismos comportamientos de validación que en CLIP, pero con default 50%
            try:
                t = float(self.local_settings.getSetting("objects_confidence_threshold"))
                if 1 <= t <= 100:
                    self.similarity_threshold = t / 100.0
                else:
                    # Fuera de rango → default 0%
                    self.similarity_threshold = 0
            except:
                # Si no hay valor o es inválido, por defecto 0%
                self.similarity_threshold = 0
        else:
            # Mismos comportamientos anteriores para CLIP
            try:
                t = float(self.local_settings.getSetting("clip_similarity_threshold"))
                if 1 <= t <= 100:
                    self.similarity_threshold = t / 100.0
                else:
                    self.similarity_threshold = 0.8 if self.functionality == "clip_img" else 0.2
            except:
                self.similarity_threshold = 0.8 if self.functionality == "clip_img" else 0.2

        self.clip_image_path = self.local_settings.getSetting("clip_image_path") or ""
        self.clip_text_query = self.local_settings.getSetting("clip_text_query") or ""
        self.server_url = "http://10.56.67.145:8000/process"
        #self.server_url = "http://localhost:8000/process"

        # Directorio temporal
        self.temp_dir = os.path.join(
            Case.getCurrentCase().getTempDirectory(),
            "Imagenes_Procesadas"
        )
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        file_log("DEBUG MODULE: Configured to send batch to {} — model={}, mode={}, threshold={}".format(
            self.server_url, self.model_flag, self.functionality, self.similarity_threshold
        ))
        self.log(Level.INFO, "Configured: url={}, model={}, mode={}, threshold={}".format(
            self.server_url, self.model_flag, self.functionality, self.similarity_threshold
        ))

    def process(self, dataSource, progressBar):
        file_log("DEBUG MODULE: process() called on dataSource")
        try:
            # 1.1) Obtenemos el objeto SleuthkitCase para hacer la consulta SQL directamente
            skCase = Case.getCurrentCase().getSleuthkitCase()

            # 1.2) Construimos la cláusula WHERE para buscar solo imágenes (.jpg, .jpeg, .png)
            #    dataSource.getId() es el data_source_obj_id en tsk_files
            dsId = dataSource.getId()
            whereClause = (
                "data_source_obj_id = %d AND ("
                "lower(name) LIKE '%%.jpg' OR lower(name) LIKE '%%.jpeg' OR lower(name) LIKE '%%.png'"
                ")" % dsId
            )

            # 1.3) Usamos findAllFilesWhere(...) para obtener un java.util.List<AbstractFile>
            imageFiles = skCase.findAllFilesWhere(whereClause)
            totalImgs = imageFiles.size()
            file_log("DEBUG MODULE: SleuthkitCase encontró %d imágenes" % totalImgs)

            # 2) Barra de progreso
            progressBar.switchToDeterminate(imageFiles.size())
            file_log("DEBUG MODULE: progressBar switchToDeterminate({})".format(imageFiles.size()))
            count = 0

            for idx in range(imageFiles.size()):
                f = imageFiles.get(idx)
                file_log("DEBUG MODULE: Loop idx={} file={}".format(idx, f.getName()))

                # 2a) Cancelacion correcta
                if self.context.isJobCancelled():
                    file_log("DEBUG MODULE: process() cancelado por el usuario en idx={}".format(idx))
                    return IngestModule.ProcessResult.OK

                name = f.getName()
                local_path = os.path.join(self.temp_dir, name)
                file_log("DEBUG MODULE: procesando imagen '{}', la copiare a '{}'".format(name, local_path))

                # 3) Copia usando ReadContentInputStream en lugar de writeToStream()
                try:
                    in_s = ReadContentInputStream(f)
                    file_log("DEBUG MODULE: abierto ReadContentInputStream para '{}'".format(name))
                    outFile = JavaFile(local_path)
                    outStream = FileOutputStream(outFile)
                    file_log("DEBUG MODULE: iniciando bucle de lectura/escritura para '{}'".format(name))
                    buf = jarray.zeros(4096, "b")
                    r = in_s.read(buf)
                    while r > 0:
                        outStream.write(buf, 0, r)
                        r = in_s.read(buf)
                    outStream.close()
                    in_s.close()
                    file_log("DEBUG MODULE: copia completada para '{}'".format(name))
                except Exception as e:
                    self.log(Level.SEVERE, "Error saving image: {}".format(e))
                    file_log("DEBUG MODULE: Exception al copiar '{}': {}".format(name, e))
                    continue

                # 4) Acumula en el lote
                self.filesProcessed += 1
                self.imageItems.append({
                    "file_obj":  f,
                    "file_name": name,
                    "path":      local_path
                })
                file_log("DEBUG MODULE: imagen '{}' añadida al lote (filesProcessed={})".format(name, self.filesProcessed))

                self.log(Level.INFO, "Image {} added to batch.".format(name))
                file_log("DEBUG MODULE: log INFO Image {} added to batch.".format(name))

                # 5) Actualiza progreso
                count += 1
                progressBar.progress(count)
                file_log("DEBUG MODULE: progressBar.progress({})".format(count))

            file_log("DEBUG MODULE: bucle terminado. total procesadas={}".format(self.filesProcessed))

        except Exception as e:
            self.log(Level.SEVERE, "Error scanning dataSource: {}".format(e))
            file_log("DEBUG MODULE: process() error en el try: {}".format(e))
            return IngestModule.ProcessResult.ERROR

        return IngestModule.ProcessResult.OK
    
    def shutDown(self):
        file_log("DEBUG MODULE: shutDown() called. filesProcessed={}".format(self.filesProcessed))
        self._send_batch_and_create_artifacts()

    def _send_batch_and_create_artifacts(self):
        file_log("DEBUG MODULE: _send_batch_and_create_artifacts() start.")
        if not self.imageItems:
            file_log("DEBUG MODULE: No hay imagenes en el lote; saliendo.")
            return

        file_log("DEBUG MODULE: filesProcessed = {}".format(self.filesProcessed))
        file_log("DEBUG MODULE: Construyendo payload inicial.")
        payload = {
            "model":  self.model_flag,
            "mode":   self.functionality,
            "images": []
        }
        file_log("DEBUG MODULE: Payload inicial: model={}, mode={}".format(
            payload["model"], payload["mode"]))

        # Añadir referencia CLIP si procede
        if self.functionality == "clip_img":
            try:
                file_log("DEBUG MODULE: Leyendo referencia CLIP_IMG de '{}'".format(self.clip_image_path))
                ref_bytes = Files.readAllBytes(Paths.get(self.clip_image_path))
                ref_b64 = Base64.getEncoder().encodeToString(ref_bytes)
                payload["reference"] = {
                    "filename": os.path.basename(self.clip_image_path),
                    "data":     ref_b64
                }
                file_log("DEBUG MODULE: Referencia CLIP_IMG añadida al payload")
            except Exception as e:
                file_log("DEBUG MODULE: Error al leer referencia CLIP_IMG: {}".format(e))
        elif self.functionality == "clip_txt":
            payload["reference"] = self.clip_text_query
            file_log("DEBUG MODULE: Referencia CLIP_TXT añadida al payload: '{}'".format(self.clip_text_query))

        # Codificar todas las imagenes
        enc = Base64.getEncoder()
        for idx, it in enumerate(self.imageItems):
            try:
                file_log("DEBUG MODULE: Codificando imagen {}/{}: '{}'".format(
                    idx+1, len(self.imageItems), it["file_name"]))
                b = Files.readAllBytes(Paths.get(it["path"]))
                img_b64 = enc.encodeToString(b)
                payload["images"].append({
                    "filename": it["file_name"],
                    "data":     img_b64
                })
            except Exception as e:
                file_log("DEBUG MODULE: Error codificando '{}': {}".format(it["file_name"], e))

        file_log("DEBUG MODULE: Payload final contiene {} imagenes".format(len(payload["images"])))

        # Enviar HTTP POST
        try:
            file_log("DEBUG MODULE: Abriendo conexion a {}".format(self.server_url))
            conn = URL(self.server_url).openConnection()
            conn.setDoOutput(True)
            conn.setRequestMethod("POST")
            conn.setRequestProperty("Content-Type", "application/json")

            file_log("DEBUG MODULE: Enviando JSON al servidor")
            w = OutputStreamWriter(conn.getOutputStream(), "UTF-8")
            w.write(json.dumps(payload))
            w.close()

            # Leer la respuesta
            file_log("DEBUG MODULE: Leyendo respuesta HTTP")
            ins = conn.getInputStream()
            resp_bytes = io.BytesIO()
            buf = jarray.zeros(4096, "b")
            r = ins.read(buf)
            while r > 0:
                resp_bytes.write(buf[:r])
                r = ins.read(buf)
            ins.close()
            conn.disconnect()
            file_log("DEBUG MODULE: Respuesta recibida ({} bytes)".format(len(resp_bytes.getvalue())))

            output = json.loads(resp_bytes.getvalue().decode('utf-8'))
            file_log("DEBUG MODULE: JSON parseado, resultados encontrados: {}".format(
                len(output.get("results", []))))
        except Exception as e:
            file_log("DEBUG MODULE: Excepcion durante HTTP POST: {}".format(e))
            return

        # Procesar resultados
        bb = Case.getCurrentCase().getServices().getBlackboard()
        tm = Case.getCurrentCase().getServices().getTagsManager()
        case_dir = Case.getCurrentCase().getTempDirectory()

        for ri, res in enumerate(output.get("results", [])):
            fname = res.get("filename")
            file_log("DEBUG MODULE: Procesando resultado {}/{}: '{}'".format(
                ri+1, len(output["results"]), fname))
            item = next((x for x in self.imageItems if x["file_name"] == fname), None)
            if not item:
                file_log("DEBUG MODULE: No se encontro item para '{}'".format(fname))
                continue
            fobj = item["file_obj"]

            # --- OBJECT DETECTION ---
            if self.functionality == "objects":
                file_log("DEBUG MODULE: Rama OBJECT DETECTION para '{}'".format(fname))
                # 🛠️ Log del umbral actual justo antes de usarlo:
                file_log("DEBUG MODULE: Current similarity_threshold = {:.3f}".format(self.similarity_threshold))
                confs = {}
                for o in res.get("objects", []):
                    cls = o.get("class_name")
                    conf = o.get("confidence", 0)
                    file_log("DEBUG MODULE: Detected {} with confidence {}".format(cls, conf))
                    if conf >= self.similarity_threshold:
                        confs.setdefault(cls, []).append(conf)
                file_log("DEBUG MODULE: Confidencias filtradas: {}".format(confs))
                if not confs:
                    file_log("DEBUG MODULE: Ninguna clase supera el umbral → fallback ImposibleClasificar")
                    confs = {"ImposibleClasificar": [1.0]}
                pct = int(self.similarity_threshold * 100)
                base = os.path.join(case_dir, "Detected_With_{}_Confianza_{}pct".format(self.model_flag, pct))
                file_log("DEBUG MODULE: Creando carpeta {}".format(base))
                if not os.path.exists(base):
                    os.makedirs(base)
                for cls, lst in confs.items():
                    fld = os.path.join(base, cls)
                    file_log("DEBUG MODULE: Creando subcarpeta {}".format(fld))
                    if not os.path.exists(fld):
                        os.makedirs(fld)
                    shutil.copy(item["path"], os.path.join(fld, fname))
                art = fobj.newArtifact(BlackboardArtifact.ARTIFACT_TYPE.TSK_INTERESTING_FILE_HIT)
                # Si es imposible clasificar, construyo un comentario especial:
                key = next(iter(confs))  # la única clave, p.ej. "ImposibleClasificar_30pct"
                if key.startswith("ImposibleClasificar"):
                    # comentario indicando modelo y umbral
                    comment_text = "Imposible Clasificar usando {} con umbral {}%".format(self.model_flag, pct)
                else:
                    avg = ", ".join("{}({:.3f})".format(c, sum(lst)/len(lst)) for c, lst in confs.items())
                    comment_text = "Detections: {}".format(avg)
                #attr = BlackboardAttribute(
                #    BlackboardAttribute.ATTRIBUTE_TYPE.TSK_SET_NAME,
                #    "AI Object Detection",
                #    "{} detecciones: {}".format(self.model_flag, avg)
                #)
                attr = BlackboardAttribute(
                    BlackboardAttribute.ATTRIBUTE_TYPE.TSK_SET_NAME,
                    "AI Object Detection",
                    "Detected With {} Confianza {}pct".format(self.model_flag, pct)
                )
                art.addAttribute(attr)
                # 2) Comentario que irá a la columna “Conclusion”
                comment_attr = BlackboardAttribute(
                    BlackboardAttribute.ATTRIBUTE_TYPE.TSK_COMMENT,
                    "AI Object Detection",
                    comment_text
                )
                art.addAttribute(comment_attr)
                bb.indexArtifact(art)
                file_log("DEBUG MODULE: Artefacto creado para '{}'".format(fname))
                for cls, lst in confs.items():
                    avg_conf = sum(lst)/len(lst)*100
                    #tag_name = "AI_{}_{}".format(self.model_flag, cls)
                    tag_name = "[{}] Detected".format(cls)
                    file_log("DEBUG MODULE: Creando tag {} con comment {:.1f}%".format(
                        tag_name, avg_conf))
                    tn = next((t for t in tm.getTagNamesInUse() if t.getDisplayName() == tag_name), None)
                    if not tn:
                        tn = tm.addTagName(tag_name)
                    if cls == "ImposibleClasificar":
                        comment = "Imposible Clasificar usando {} con umbral {}%".format(self.model_flag, pct)
                    else:
                        comment = "{:.1f}% - Confianza con {}".format(avg_conf, self.model_flag)
                    tm.addContentTag(fobj, tn, comment)

            # --- CLIP_IMG ---
            elif self.functionality == "clip_img":
                file_log("DEBUG MODULE: Rama CLIP_IMG para '{}'".format(fname))
                sim = res.get("similarity", 0)
                file_log("DEBUG MODULE: Similarity score = {}".format(sim))
                if sim >= self.similarity_threshold:
                    pct = int(self.similarity_threshold * 100)
                    ref_base = os.path.splitext(os.path.basename(self.clip_image_path))[0]
                    dir_ = os.path.join(case_dir, "Searched_Image_{}_With_CLIP_IMG_Confianza_{}pct".format(ref_base,pct))
                    file_log("DEBUG MODULE: Creando carpeta {}".format(dir_))
                    if not os.path.exists(dir_):
                        os.makedirs(dir_)
                    shutil.copy(item["path"], os.path.join(dir_, fname))
                    art = fobj.newArtifact(BlackboardArtifact.ARTIFACT_TYPE.TSK_INTERESTING_FILE_HIT)
                    #attr = BlackboardAttribute(
                    #    BlackboardAttribute.ATTRIBUTE_TYPE.TSK_SET_NAME,
                    #    "AI Object Search",
                    #    "CLIP_IMG similitud {:.1f}%".format(sim * 100)
                    #)
                    attr = BlackboardAttribute(
                        BlackboardAttribute.ATTRIBUTE_TYPE.TSK_SET_NAME,
                        "AI Object Detection", #AI Object Search?
                        "Searched With CLIP_IMG Confianza {}pct".format(pct)
                    )
                    art.addAttribute(attr)
                    comment_attr = BlackboardAttribute(
                        BlackboardAttribute.ATTRIBUTE_TYPE.TSK_COMMENT,
                        "AI Object Search",
                        "Similarity: {:.1f}% [RefImage: '{}']".format(sim * 100, ref_base)
                    )
                    art.addAttribute(comment_attr)
                    bb.indexArtifact(art)
                    tag_name = "{} [RefImage: '{}']".format(self.model_flag, ref_base)
                    tn = next((t for t in tm.getTagNamesInUse() if t.getDisplayName() == tag_name), None)
                    if not tn:
                        tn = tm.addTagName(tag_name)
                    tm.addContentTag(fobj, tn, "{:.1f}% - Similitud con {}".format(sim * 100, self.model_flag))

            # --- CLIP_TXT ---
            elif self.functionality == "clip_txt":
                file_log("DEBUG MODULE: Rama CLIP_TXT para '{}'".format(fname))
                sim = res.get("similarity", 0)
                file_log("DEBUG MODULE: Similarity score = {}".format(sim))
                if sim >= self.similarity_threshold:
                    pct = int(self.similarity_threshold * 100)
                    txt_base = self.clip_text_query.strip().replace(" ", "_")
                    dir_ = os.path.join(case_dir, "Searched_{}_With_CLIP_TXT_Confianza_{}pct".format(txt_base,pct))
                    file_log("DEBUG MODULE: Creando carpeta {}".format(dir_))
                    if not os.path.exists(dir_):
                        os.makedirs(dir_)
                    shutil.copy(item["path"], os.path.join(dir_, fname))
                    art = fobj.newArtifact(BlackboardArtifact.ARTIFACT_TYPE.TSK_INTERESTING_FILE_HIT)
                    #attr = BlackboardAttribute(
                    #    BlackboardAttribute.ATTRIBUTE_TYPE.TSK_SET_NAME,
                    #    "AI Object Detection",
                    #    "CLIP_TXT similitud {:.1f}%".format(sim * 100)
                    #)
                    attr = BlackboardAttribute(
                        BlackboardAttribute.ATTRIBUTE_TYPE.TSK_SET_NAME,
                        "AI Object Detection", #AI Object Search?
                        "Searched With CLIP_TXT Confianza {}pct".format(pct)
                    )
                    art.addAttribute(attr)
                    comment_attr = BlackboardAttribute(
                        BlackboardAttribute.ATTRIBUTE_TYPE.TSK_COMMENT,
                        "AI Object Search",
                        "Similarity: {:.1f}% [RefText: '{}']".format(sim * 100, txt_base)
                    )
                    art.addAttribute(comment_attr)
                    bb.indexArtifact(art)
                    tag_name = "{} [RefText: '{}']".format(self.model_flag, txt_base)
                    tn = next((t for t in tm.getTagNamesInUse() if t.getDisplayName() == tag_name), None)
                    if not tn:
                        tn = tm.addTagName(tag_name)
                    tm.addContentTag(fobj, tn, "{:.1f}% - Similitud con {}".format(sim * 100, self.model_flag))

        # Limpieza final
        self.imageItems = []
        self.filesProcessed = 0
        file_log("DEBUG MODULE: Batch processing complete.")
