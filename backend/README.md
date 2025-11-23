# Sistema de Reconocimiento Facial con OpenCV

Sistema simple de reconocimiento facial usando OpenCV que entrena modelos desde video y reconoce caras en tiempo real desde streaming.

## Requisitos

### Hardware
- Cámara web USB (para reconocimiento)
- Video de entrenamiento (para entrenar el modelo)

### Software
- Python 3.7 o superior
- OpenCV con contrib (para reconocedores faciales)

## Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

**Nota:** `opencv-contrib-python` es necesario para los reconocedores faciales (cv2.face).

### 2. Verificar instalación

```python
import cv2
print(cv2.__version__)
print(hasattr(cv2, 'face'))  # Debe ser True
```

## Uso

### Entrenar Modelo desde Video

El script `train_model.py` permite entrenar un modelo de reconocimiento facial a partir de un video.

```bash
python train_model.py video.mp4 --persons Persona1 Persona2 Persona3 --model face_model.xml
```

**Parámetros:**
- `video`: Ruta al archivo de video para entrenamiento
- `--persons`: Nombres de las personas (una para cada número del 1 al 9)
- `--model`: Nombre del archivo de modelo a generar (default: `face_model.xml`)

**Proceso de entrenamiento:**

1. Se reproduce el video y detecta caras automáticamente
2. Presiona una tecla numérica (1-9) cuando veas la cara de una persona para etiquetarla:
   - Tecla `1` → Primera persona de la lista
   - Tecla `2` → Segunda persona de la lista
   - etc.
3. El sistema extrae múltiples frames de cada cara detectada
4. Presiona `q` para finalizar y guardar el modelo

**Ejemplo:**

```bash
python train_model.py training_video.mp4 --persons Juan Maria Pedro --model mi_modelo.xml
```

Durante el entrenamiento:
- Presiona `1` cuando veas a Juan
- Presiona `2` cuando veas a María
- Presiona `3` cuando veas a Pedro
- Presiona `q` para finalizar

### Reconocimiento en Tiempo Real

El script `recognize.py` carga el modelo entrenado y reconoce caras en tiempo real desde la cámara.

```bash
python recognize.py --model face_model.xml --camera 0 --threshold 100
```

**Parámetros:**
- `--model`: Ruta al archivo de modelo XML (default: `face_model.xml`)
- `--camera`: Índice de la cámara (default: 0)
- `--threshold`: Umbral de confianza - menor es más confiable (default: 100, rango típico: 50-150)
- `--debug`: Modo debug para ver información detallada

**Controles durante reconocimiento:**
- `q`: Salir
- `+` o `=`: Aumentar umbral de confianza (hace el reconocimiento más estricto)
- `-`: Disminuir umbral de confianza (hace el reconocimiento más permisivo)
- `d`: Activar/desactivar modo debug

**Ejemplo:**

```bash
# Reconocimiento básico
python recognize.py --model mi_modelo.xml --camera 0

# Con debug para ver valores de confianza
python recognize.py --model mi_modelo.xml --camera 0 --debug

# Con threshold personalizado
python recognize.py --model mi_modelo.xml --camera 0 --threshold 120
```

## Archivos Generados

Después del entrenamiento se generan dos archivos:

1. **`face_model.xml`** (o el nombre especificado): Modelo entrenado
2. **`face_model_labels.txt`** (o `[modelo]_labels.txt`): Mapeo de IDs a nombres

El archivo de labels contiene:
```
1:Juan
2:Maria
3:Pedro
```

## Estructura del Proyecto

```
backend/
├── train_model.py          # Script para entrenar modelo desde video
├── recognize.py            # Script para reconocimiento en streaming
├── face_model.xml          # Modelo entrenado (generado)
├── face_model_labels.txt   # Mapeo de labels (generado)
├── requirements.txt        # Dependencias
└── README.md              # Este archivo
```

## Diagnosticar Problemas de Reconocimiento

### Verificar el Modelo

Primero, verifica que el modelo se haya entrenado correctamente:

```bash
python test_model.py --model face_model.xml
```

Este script verifica:
- Que el modelo existe y se puede cargar
- Que el mapeo de labels es correcto
- Muestra los labels disponibles

### Si Siempre Muestra "Unknown"

1. **Verifica el threshold**: El threshold por defecto es 100. Prueba:
   ```bash
   python recognize.py --threshold 150  # Más permisivo
   python recognize.py --threshold 80   # Más estricto
   ```

2. **Usa modo debug** para ver los valores de confianza:
   ```bash
   python recognize.py --debug
   ```
   Presiona `d` durante el reconocimiento para activar debug en la consola.

3. **Ajusta el threshold en tiempo real**:
   - Presiona `+` para aumentar (hacer más permisivo)
   - Presiona `-` para disminuir (hacer más estricto)
   - Observa cómo cambia el reconocimiento

4. **Verifica condiciones de iluminación**: 
   - La iluminación debe ser similar a la del entrenamiento
   - Evita sombras fuertes en la cara
   - Usa iluminación frontal si es posible

5. **Reentrena con más datos**:
   - Extrae más frames durante el entrenamiento
   - Asegúrate de tener al menos 10-20 frames por persona
   - Incluye diferentes ángulos y expresiones

### Valores Típicos de Confianza LBPH

- **0-50**: Excelente coincidencia (muy confiable)
- **50-80**: Buena coincidencia (confiable)
- **80-100**: Coincidencia aceptable
- **100-150**: Coincidencia débil (puede ser confiable si el threshold es alto)
- **>150**: Coincidencia muy débil (probablemente incorrecta)

Si ves valores de confianza >150 constantemente, el modelo necesita más entrenamiento.

## Tips para Mejor Rendimiento

### Entrenamiento

1. **Buena iluminación**: Asegúrate de tener buena iluminación en el video de entrenamiento
2. **Múltiples ángulos**: Intenta capturar diferentes ángulos y expresiones faciales
3. **Calidad del video**: Usa videos con buena resolución y sin mucho movimiento de cámara
4. **Tiempo suficiente**: Permite que el sistema extraiga suficientes frames de cada persona

### Reconocimiento

1. **Ajustar threshold**: Si hay muchos falsos positivos, aumenta el threshold
2. **Iluminación consistente**: Usa iluminación similar a la del entrenamiento
3. **Distancia adecuada**: Mantén una distancia similar a la del entrenamiento

## Solución de Problemas

### Error: "Model file not found"

Asegúrate de haber entrenado un modelo primero usando `train_model.py`.

### Error: "Could not open camera"

1. Verifica que la cámara esté conectada
2. Prueba con diferentes índices: `--camera 1`, `--camera 2`, etc.
3. Cierra otras aplicaciones que puedan estar usando la cámara

### No se detectan caras

1. Verifica que la cámara esté funcionando correctamente
2. Mejora la iluminación
3. Asegúrate de estar mirando directamente a la cámara

### Reconocimiento incorrecto o siempre "Unknown"

1. **Ajusta el threshold**: Prueba valores entre 50-150
   - Si siempre muestra "Unknown", aumenta el threshold (presiona `+`)
   - Si muestra nombres incorrectos, disminuye el threshold (presiona `-`)

2. **Verifica valores de confianza**: Usa `--debug` para ver los valores reales
   ```bash
   python recognize.py --debug
   ```
   Observa los valores en la consola. Si son >150, necesitas más entrenamiento.

3. **Reentrena con más datos**: 
   - Extrae más frames durante el entrenamiento
   - Al menos 10-20 frames por persona
   - Incluye diferentes ángulos y expresiones

4. **Condiciones similares**:
   - Iluminación similar entre entrenamiento y reconocimiento
   - Distancia similar a la cámara
   - Ángulos similares

5. **Prueba el modelo**:
   ```bash
   python test_model.py --model face_model.xml
   ```
   Verifica que el modelo y los labels se carguen correctamente

### Error al importar cv2.face

Asegúrate de tener instalado `opencv-contrib-python`:
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

## Algoritmo Utilizado

Este sistema usa **LBPH (Local Binary Patterns Histograms) Face Recognizer** de OpenCV, que es:

- Robusto ante cambios de iluminación
- Rápido en tiempo de ejecución
- Efectivo para pequeñas bases de datos
- Guarda modelos en formato XML/YAML

## Limitaciones

- Máximo 9 personas diferentes (limitado por teclas numéricas 1-9)
- Requiere buen etiquetado manual durante el entrenamiento
- Funciona mejor con iluminación consistente
- Puede tener dificultades con ángulos muy diferentes al entrenamiento
