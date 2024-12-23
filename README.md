# Uso de Tecnologías para Procesamiento de Documentos Legales

Este documento describe cómo usar herramientas y tecnologías de IA para procesar documentos legales, extraer información clave y optimizar operaciones en el ámbito jurídico. Se incluyen ejemplos prácticos para cada tecnología.

---

## **Procesamiento de Lenguaje Natural (NLP)**

### **Uso de spaCy para Reconocimiento de Entidades Nombradas (NER)**

```python
import spacy

# Cargar el modelo en español de spaCy
nlp = spacy.load('es_core_news_sm')

# Texto a analizar
texto = "El contrato firmado entre Juan Pérez y María López en Calle Falsa 123 el 5 de enero de 2023 establece..."

# Procesar el texto
doc = nlp(texto)

# Extraer entidades
for ent in doc.ents:
    print(f"Entidad: {ent.text}, Tipo: {ent.label_}")
```

**Resultados esperados:**
- Entidad: Juan Pérez, Tipo: PER (Persona)
- Entidad: María López, Tipo: PER (Persona)
- Entidad: Calle Falsa 123, Tipo: LOC (Lugar)
- Entidad: 5 de enero de 2023, Tipo: DATE (Fecha)

### **Uso de Hugging Face Transformers para Modelos Avanzados**

```python
from transformers import pipeline

# Crear un pipeline para reconocimiento de entidades
modelo = pipeline("ner", model="dslim/bert-base-NER")

# Texto a analizar
texto = "Juan Pérez firmó el contrato el 5 de enero de 2023 en Calle Falsa 123."

# Realizar análisis
resultado = modelo(texto)
for entidad in resultado:
    print(entidad)
```

**Resultados esperados:**
- {'word': 'Juan', 'entity': 'B-PER', 'score': 0.99}
- {'word': 'Pérez', 'entity': 'I-PER', 'score': 0.98}
- {'word': '5 de enero de 2023', 'entity': 'DATE', 'score': 0.95}

---

## **Reconocimiento Óptico de Caracteres (OCR)**

### **Uso de Tesseract OCR**

```python
from pytesseract import image_to_string
from PIL import Image

# Abrir una imagen de un documento escaneado
imagen = Image.open("documento_escaneado.png")

# Extraer texto
texto = image_to_string(imagen, lang='spa')
print(texto)
```

### **Mejorar la Imagen con OpenCV**

```python
import cv2
from pytesseract import image_to_string

# Leer la imagen
imagen = cv2.imread("documento_escaneado.png")

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar binarización
umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Guardar la imagen procesada
cv2.imwrite("procesada.png", umbral)

# Extraer texto con Tesseract
texto = image_to_string("procesada.png", lang='spa')
print(texto)
```

---

## **Almacenamiento y Búsqueda**

### **Guardar Datos con MongoDB**

```python
from pymongo import MongoClient

# Conectar a la base de datos
cliente = MongoClient("mongodb://localhost:27017/")
db = cliente["documentos_legales"]

# Insertar un documento
documento = {
    "nombre": "Contrato de Arrendamiento",
    "texto": "El presente contrato entre Juan Pérez y María López...",
    "fecha": "2023-01-05"
}
db.contratos.insert_one(documento)
print("Documento insertado")
```

### **Búsqueda con Elasticsearch**

```python
from elasticsearch import Elasticsearch

# Conectar al servidor de Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Indexar un documento
es.index(index="contratos", id=1, body={
    "titulo": "Contrato de Arrendamiento",
    "contenido": "El presente contrato entre Juan Pérez y María López..."
})

# Buscar documentos relacionados
resultado = es.search(index="contratos", body={
    "query": {"match": {"contenido": "arrendamiento"}}
})
print(resultado)
```

---

## **Construcción de una API**

### **Backend con FastAPI**

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/procesar/")
async def procesar_documento(texto: str):
    # Procesar texto con spaCy
    import spacy
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(texto)
    entidades = [{"texto": ent.text, "tipo": ent.label_} for ent in doc.ents]
    return {"entidades": entidades}
```

Ejecuta esta API y utiliza herramientas como Postman para enviar texto y recibir entidades extraídas.

---
