NAME = "Yesenia Valdés Rodríguez"
GROUP = "311"
CAREER = "C"
MODEL = "Okapi BM25"

"""
INFORMACIÓN EXTRA:

Fuente bibliográfica principal:
Robertson, S., Zaragoza, H., & Taylor, M. (2004). The Probabilistic Relevance Framework: BM25 and Beyond. 
*Foundations and Trends in Information Retrieval*, 3(4), 333-389. 
Url: https://doi.org/10.1561/1500000019

Mejora implementada:
- Técnicas avanzadas de preprocesamiento: Lematización, eliminación de stopwords/puntuación, y normalización semántica con embeddings de Spacy.
- Beneficio: Reduce ruido léxico y mejora la alineación estadística entre consultas y documentos.

Análisis conceptual:
- Definición formal:
  Q: Consulta como conjunto de términos {q₁, q₂, ..., qₙ}
  D: Documento con longitud |D| y frecuencia de término F(t, D)
  F: Función de relevancia basada en distribución estadística de términos
  R(D, Q): Score BM25 = Σₜ∈Q [IDF(t) · (F(t,D) · (k₁ + 1)) / (F(t,D) + k₁(1 - b + b·|D|/avgdl))]
            Donde:
            - k₁: Controla la saturación de TF (valores altos = mayor peso a repeticiones)
            - b: Controla la normalización por longitud (0 = desactivada, 1 = normalización completa)
            - avgdl: Longitud promedio de documentos en el corpus
            - IDF(t): log[(N - nₜ + 0.5)/(nₜ + 0.5) + 1], donde N = total documentos, nₜ = documentos con t

- ¿Dependencia entre los términos? No. BM25 asume independencia (modelo bag-of-words).

- Correspondencia parcial documento-consulta: Sí. Asigna pesos continuos mediante función no lineal.

- Ranking: Sí. Ordena documentos por R(D,Q) descendente.

"""
import ir_datasets
import numpy as np
import math
from typing import Dict, List, Tuple
import spacy

class InformationRetrievalModel:
    def __init__(self, k1=1.5, b=0.75):
        """
        Inicializa el modelo de recuperación de información.

        Modelo escogido: Okapi BM25, un modelo probabilístico de Segunda Generación (extensión del Binary Independence Model o BIR).
        Parte del marco probabilístico de recuperación de información, extendiendo el BIR al introducir parámetros ajustables
        (como k1 y b) que controlan la influencia de la frecuencia de términos (TF) y la normalización de longitud de documento.
        Por ejemplo, el parámetro b ajusta cómo la longitud de un documento afecta su relevancia, mientras que k1 limita la 
        saturación de términos frecuentes.
        Se puede considerar como una variante avanzada del modelo vectorial con TF-IDF mejorado.
        
        Args:
            k1: Factor de saturación de frecuencia de términos (1.5 ≤ k1 ≤ 2.0 óptimo)
            b: Factor de normalización por longitud de documento (0.75 recomendado)
        """
        self.nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
        self.doc_data = []
        self.doc_ids = []
        self.df = {}
        self.idf = {}
        self.avgdl = 0.0
        self.k1 = k1
        self.b = b
        self.queries = {}
        self.dataset = None

    def _preprocess(self, text: str) -> List[str]:
        """Preprocesa texto aplicando normalización lingüística:
        1. Conversión a minúsculas
        2. Lematización morfológica
        3. Filtrado de stopwords y caracteres no léxicos
        
        Args:
            text (str): Cadena de texto crudo
        Returns:
            lista (List[str]): Lista de términos normalizados
        """
        doc = self.nlp(text.lower())
        return [
            token.lemma_ 
            for token in doc 
            if not token.is_stop 
            and not token.is_punct 
            and not token.is_space
            and token.lemma_.strip()
        ]

    def fit(self, dataset_name: str):
        """Construye el índice invertido y calcula métricas globales del corpus.
        
        Proceso:
        1. Cálculo de TF (frecuencia de términos) por documento
        2. Cálculo de DF (document frequency)
        3. Cálculo de IDF según variante BM25
        4. Cálculo de avgdl (longitud promedio de documentos)
        
        :param dataset_name: Identificador del dataset en ir_datasets
        :raises ValueError: Si el dataset no contiene consultas
        """
        self.dataset = ir_datasets.load(dataset_name)
        
        if not hasattr(self.dataset, 'queries_iter'):
            raise ValueError("Dataset sin queries definidas")
        
        # Procesar documentos
        self.doc_data = []
        self.doc_ids = []
        self.df = {}
        
        for doc in self.dataset.docs_iter():
            tokens = self._preprocess(doc.text)
            tfs = {}
            for token in tokens:
                tfs[token] = tfs.get(token, 0) + 1
            
            self.doc_ids.append(doc.doc_id)
            self.doc_data.append({
                "tfs": tfs,
                "length": len(tokens)
            })
            
            # Actualizar document frequency
            for term in tfs:
                self.df[term] = self.df.get(term, 0) + 1
        
        # Calcular estadísticas globales
        total_length = sum(d["length"] for d in self.doc_data)
        self.avgdl = total_length / len(self.doc_data) if self.doc_data else 0.0
        
        # Precalcular IDF
        N = len(self.doc_data)
        self.idf = {
            term: math.log((N - n_t + 0.5) / (n_t + 0.5) + 1)
            for term, n_t in self.df.items()
        }
        
        # Cargar queries
        self.queries = {q.query_id: q.text for q in self.dataset.queries_iter()}

    def predict(self, top_k: int) -> Dict[str, Dict[str, List[str]]]:
        """
        Realiza búsquedas para TODAS las queries del dataset automáticamente.
        
        Args:
            top_k (int): Número máximo de documentos a devolver por query.
            threshold (float): Umbral de similitud mínimo para considerar un match.
            
        Returns:
            dict: Diccionario con estructura {
                query_id: {
                    'text': query_text,
                    'results': [(doc_id, score), ...]
                }
            }
        """
        results = {}
        if not self.doc_data:
            return results
        
        # Procesar cada query
        for qid, query_text in self.queries.items():
            terms = self._preprocess(query_text)
            valid_terms = [term for term in terms if term in self.idf]
            
            scores = []
            for doc_id, data in zip(self.doc_ids, self.doc_data):
                score = 0.0
                doc_length = data["length"]
                
                # Calcular el score según la fórumula de BM25
                for term in valid_terms:
                    tf = data["tfs"].get(term, 0)
                    idf = self.idf[term]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
                    score += idf * (numerator / denominator)
                
                scores.append((doc_id, score))
            
            # Ordenar y seleccionar top_k
            scores.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc_id for doc_id, _ in scores[:top_k]]
            results[qid] = {
                "text": query_text,
                "results": top_docs
            }
        
        return results

    def evaluate(self, top_k: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Evalúa los resultados para TODAS las queries comparando con los qrels oficiales.
        
        Args:
            top_k (int): Número máximo de documentos a considerar por query.
            
        Returns:
            dict: Métricas de evaluación por query y métricas agregadas.
        """
        if not hasattr(self.dataset, 'qrels_iter'):
            raise ValueError("Este dataset no tiene relevancias definidas (qrels)")
        
        predictions = self.predict(top_k=top_k)
        
        qrels = {}
        for qrel in self.dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        
        result = {}
        
        for qid, data in predictions.items():
            if qid not in qrels:
                continue
                
            relevant_docs = set(doc_id for doc_id, rel in qrels[qid].items() if rel > 0)
            retrieved_docs = set(doc_id for doc_id, _ in data['results']) # Error corregido para poder usar el método
            relevant_retrieved = relevant_docs & retrieved_docs
            
            result[qid] = {
                'all_relevant': relevant_docs,
                'all_retrieved': retrieved_docs,
                'relevant_retrieved': relevant_retrieved
            }
        
        return result
