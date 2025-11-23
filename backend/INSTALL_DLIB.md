# Instalación de dlib en Windows

Para instalar `dlib` y `face-recognition` en Windows, necesitas **CMake** primero.

## Opción 1: Instalar CMake manualmente (Recomendado)

1. Descarga CMake desde: https://cmake.org/download/
2. Durante la instalación, asegúrate de marcar "Add CMake to system PATH"
3. Reinicia tu terminal/PowerShell
4. Verifica la instalación:
   ```bash
   cmake --version
   ```
5. Luego instala dlib y face-recognition:
   ```bash
   cd backend
   .\venv\Scripts\Activate.ps1
   pip install dlib face-recognition
   ```

## Opción 2: Usar Chocolatey (si lo tienes instalado)

```bash
choco install cmake
```

Luego:
```bash
cd backend
.\venv\Scripts\Activate.ps1
pip install dlib face-recognition
```

## Opción 3: Usar conda (si usas Anaconda/Miniconda)

```bash
conda install -c conda-forge cmake dlib
pip install face-recognition
```

## Nota Importante

Si no puedes instalar CMake por ahora, la aplicación **puede iniciarse** pero el reconocimiento facial no funcionará. Solo podrás ver el stream de video sin reconocimiento.

