# app/etl/loader.py
"""Fase L (Load): Carga de datos transformados a la base de datos analítica."""

import pandas as pd
from sqlalchemy import text
from datetime import datetime
from app.database.connection import analytics_engine
from app.database.models import UserFeatureVector


class DataLoader:
    """Clase responsable de cargar los datos transformados a la DB Analítica."""
    
    def __init__(self):
        self.engine = analytics_engine
        self.table_name = UserFeatureVector.__tablename__
    
    def truncate_table(self):
        """Elimina todos los registros existentes antes de cargar nuevos."""
        print("🗑️ Truncando tabla existente...")
        with self.engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {self.table_name} RESTART IDENTITY;"))
    
    def load_dataframe(self, df: pd.DataFrame):
        """Carga el DataFrame a la base de datos."""
        print(f"💾 Cargando {len(df)} registros a la DB Analítica...")
        
        # Agregar fecha de extracción
        df = df.copy()
        df['extraction_date'] = datetime.utcnow()
        
        # Usar pandas to_sql para carga masiva eficiente
        df.to_sql(
            name=self.table_name,
            con=self.engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )
    
    def run_load(self, df: pd.DataFrame, truncate_before: bool = True) -> int:
        """Ejecuta el proceso completo de carga."""
        print("📤 Iniciando Fase L: Carga de datos...")
        
        if df.empty:
            print("   ⚠️ No hay datos para cargar.")
            return 0
        
        if truncate_before:
            self.truncate_table()
        
        self.load_dataframe(df)
        
        print(f"   ✅ Carga completa: {len(df)} registros insertados")
        
        return len(df)
