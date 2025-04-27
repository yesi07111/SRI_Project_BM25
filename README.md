# 📚 Proyecto de Recuperación de Información con Okapi BM25 
## 🚀 Descripción General  
Sistema de recuperación de información implementando el modelo probabilístico **BM25** con mejoras en preprocesamiento lingüístico. Evalúa relevancia documento-consulta mediante distribución estadística de términos con parámetros ajustables (`k₁`, `b`).  

Características clave:  
- ✅ Lematización avanzada con Spacy  
- ✅ Normalización semántica mediante embeddings  
- ✅ Cálculo automático de métricas TF/IDF  
- ✅ Evaluación integrada contra qrels oficiales

## 🔧 Componentes Técnicos  
### 🧠 Modelo Matemático  
```python
Score(D,Q) = Σₜ∈Q [IDF(t) · (F(t,D)·(k₁+1))/(F(t,D)+k₁(1 - b + b·|D|/avgdl))]
```
Donde:
- **IDF(t)** (*Inverse Document Frequency*):  
  `log[(N - nₜ + 0.5)/(nₜ + 0.5) + 1]`  
  - *N*: Total de documentos en el corpus  
  - *nₜ*: Documentos que contienen el término *t*  
  - **Función**: Penaliza términos comunes y premia los raros (ej: "el" vs "neurociencia").  

- **F(t,D)** (*Term Frequency*):  
  - Frecuencia cruda del término *t* en el documento *D*.  

- **k₁**:  
  - Controla la **saturación de TF** (ej: si k₁→∞, TF no se satura → repeticiones extremas ganan peso).  
  - Valor típico: 1.2-2.0. Alto k₁ = mayor influencia de repeticiones.  

- **b**:  
  - Controla la **normalización por longitud de documento** (0 = desactivada, 1 = máxima normalización).  
  - **|D|/avgdl**: Ratio entre longitud del documento y longitud promedio del corpus.  

- **Denominador**:  
  `F(t,D) + k₁(1 - b + b·|D|/avgdl)`  
  - Equilibra TF con penalización por documentos largos (evita bias hacia textos extensos).  

### 📌 Marco de Relevancia Probabilística (PRF)  
- **Hipótesis central**:  
  "La relevancia de un documento *D* a una consulta *Q* puede estimarse mediante la distribución estadística de sus términos".  

- **Supuestos clave**:  
  1. **Independencia entre términos**: BM25 asume términos no correlacionados (modelo "bag-of-words").  
  2. **Relevancia binaria**: Documentos son relevantes o no (aunque el score es continuo).  
  3. **Distribución no uniforme**: Términos relevantes tienden a agruparse en documentos relevantes.  

### 🔄 Dinámica TF-IDF Mejorada  
- **TF clásico**: Lineal (ej: 5 repeticiones = 5× peso).  
- **TF en BM25**: **Función de saturación** (ej: 5 repeticiones ≠ 5×, sino log(5) ≈ 1.6×).  
  - **Ventaja**: Mitiga spam de repeticiones (ej: "viagra viagra viagra...").  

### 🛠️ Pipeline de Procesamiento  
1. **Preprocesamiento**:  
   - 🔠 Conversión a minúsculas  
   - 🔍 Lematización morfológica  
   - 🗑️ Filtrado stopwords/puntuación  
2. **Indexado**:  
   - 📊 Cálculo TF-IDF mejorado  
   - 📐 Normalización por avgdl  
3. **Búsqueda**:  
   - 🔎 Cálculo scores BM25  
   - 🏆 Ranking automático Top-K

## 📊 Métricas de Evaluación  
```python
{
  'all_relevant': Set[doc_ids],    # Documentos realmente relevantes
  'all_retrieved': Set[doc_ids],   # Documentos recuperados
  'relevant_retrieved': Set[doc_ids]  # Intersección relevante
}
```

## 🌱 Evolución Histórica del Modelo  
### 1. Modelo Booleano (1° Generación)  
- Relevancia binaria: Documentos cumplen operadores lógicos (AND/OR/NOT).  
- **Limitación**: Sin ranking ni ponderación.  

### 2. Modelo Vectorial (TF-IDF)  
- Introduce ranking continuo basado en similitud coseno.  
- **Problema**: TF lineal y normalización rígida.  

### 3. Binary Independence Model (BIM)  
- Primer modelo probabilístico. Usa P(relevancia | términos).  
- **Fallo**: Ignora TF y longitud de documentos.  

### 4. BM25 (Okapi, 1994)  
- **Innovaciones**:  
  - Saturación no lineal de TF.  
  - Normalización adaptable por longitud (parámetro *b*).  
  - IDF robusto contra términos ultra-frecuentes.
 
## 💪 Ventajas Cualitativas de BM25  
### 🛡️ Robustez Operativa  
- **Adaptabilidad**: Parámetros *k₁* y *b* permiten ajuste fino a dominios específicos (ej: *b=0.8* para blogs largos vs *b=0.3* para tuits).  

### 🧩 Balance Léxico-Estructural  
- **Evita overfitting**:  
  - TF saturado → Reduce impacto de spam léxico.  
  - Normalización de longitud → Neutraliza ventajas artificiales de documentos largos.  

### ⚡ Eficiencia Computacional  
- **Costo O(n)**: Escala linealmente con tamaño del corpus (vs modelos neuronales O(n²)).  

### 🌍 Generalización Empírica  
- **Dominio-agnóstico**: Funciona bien en corpus heterogéneos (científicos, legales, web) sin reentrenamiento.

## ⚠️ Limitaciones Conceptuales  
- **Ceguera semántica**: Ignora sinónimos y relaciones contextuales (ej: "auto" ≠ "coche").  
- **Estaticidad**: No aprende de interacciones usuario (requiere reindexar para actualizar relevancia).  
- **Dependencia de preprocesamiento**: La calidad del lematizador/stopwords afecta resultados drásticamente.  

## 📚 Referencia Bibliográfica  
```bibtex
@article{robertson2004probabilistic,
  title={The Probabilistic Relevance Framework: BM25 and Beyond},
  author={Robertson, Stephen and Zaragoza, Hugo and Taylor, Michael},
  journal={Foundations and Trends in Information Retrieval},
  volume={3},
  number={4},
  pages={333--389},
  year={2004},
  doi={10.1561/1500000019}
}
```


## 💻 Uso Básico  
```python
# Inicialización
model = InformationRetrievalModel(k1=1.5, b=0.75)
model.fit('trec-robust04')  # Carga dataset

# Búsqueda y evaluación
results = model.predict(top_k=10)
metrics = model.evaluate()
```
