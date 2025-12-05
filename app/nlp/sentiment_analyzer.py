# app/nlp/sentiment_analyzer.py
"""Análisis de sentimiento usando modelos Transformer en español."""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from typing import List, Optional
from app.config import settings


class SentimentAnalyzer:
    """Analizador de sentimiento usando modelos Transformer en español."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.NLP_MODEL_NAME
        self.device = 0 if torch.cuda.is_available() else -1
        self._pipeline = None
        self._is_loaded = False
    
    @property
    def sentiment_pipeline(self):
        """Inicialización lazy del pipeline (solo cuando se necesita)."""
        if self._pipeline is None:
            print(f"🔄 Cargando modelo NLP: {self.model_name}...")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    return_all_scores=True,
                    truncation=True,
                    max_length=512
                )
                
                self._is_loaded = True
                print("   ✅ Modelo NLP cargado correctamente")
                
            except Exception as e:
                print(f"   ❌ Error cargando modelo NLP: {e}")
                self._pipeline = None
                self._is_loaded = False
        
        return self._pipeline
    
    def is_available(self) -> bool:
        """Verifica si el analizador está disponible."""
        return self.sentiment_pipeline is not None
    
    def analyze_text(self, text: str) -> float:
        """
        Analiza el sentimiento de un texto individual.
        Retorna la probabilidad de sentimiento negativo (0.0 - 1.0).
        """
        if not text or len(text.strip()) < 3:
            return 0.0
        
        if not self.is_available():
            return 0.0
        
        try:
            # Truncar texto muy largo
            text = text[:512]
            
            result = self.sentiment_pipeline(text)[0]
            
            # Buscar el score de la etiqueta negativa
            neg_score = next(
                (item['score'] for item in result 
                 if item['label'].upper() in ['NEG', 'NEGATIVE', 'LABEL_0']),
                0.0
            )
            
            return neg_score
            
        except Exception as e:
            print(f"   ⚠️ Error analizando texto: {e}")
            return 0.0
    
    def calculate_negativity_index(self, texts: List[str]) -> float:
        """
        Calcula el índice de negatividad promedio para una lista de textos.
        Retorna un valor entre 0.0 y 1.0.
        """
        if not texts or len(texts) == 0:
            return 0.0
        
        if not self.is_available():
            return 0.0
        
        scores = []
        for text in texts:
            score = self.analyze_text(text)
            if score > 0:
                scores.append(score)
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def batch_analyze(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """
        Analiza múltiples textos en lotes para mayor eficiencia.
        """
        if not self.is_available():
            return [0.0] * len(texts)
        
        scores = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t[:512] if t else "" for t in batch]
            
            try:
                results = self.sentiment_pipeline(batch)
                
                for result in results:
                    neg_score = next(
                        (item['score'] for item in result 
                         if item['label'].upper() in ['NEG', 'NEGATIVE', 'LABEL_0']),
                        0.0
                    )
                    scores.append(neg_score)
                    
            except Exception as e:
                print(f"   ⚠️ Error en lote: {e}")
                scores.extend([0.0] * len(batch))
        
        return scores
