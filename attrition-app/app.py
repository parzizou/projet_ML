"""
🎯 Application de Prédiction d'Attrition des Employés
====================================================
Application FastAPI pour prédire l'attrition des employés avec interface RH.
"""

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import json
import io
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# 🔧 MODIFIEZ CES CHEMINS SELON VOTRE CONFIGURATION
MODEL_PATH = os.getenv("MODEL_PATH", "models/attrition_model.joblib")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/attrition_preprocessor.joblib")
METADATA_PATH = os.getenv("METADATA_PATH", "models/attrition_metadata.joblib")

# ============================================================================
# APPLICATION FASTAPI
# ============================================================================

app = FastAPI(
    title="🎯 Prédiction Attrition RH",
    description="Application de prédiction de l'attrition des employés pour les RH",
    version="1.0.0"
)

# Templates et fichiers statiques
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================================
# CHARGEMENT DU MODÈLE
# ============================================================================

# Variables globales pour le modèle
model = None
preprocessor = None
metadata = None
imputation_values = None  # Valeurs pour l'imputation (médianes/modes)
optimal_threshold = 0.5  # Seuil de décision par défaut

def load_models():
    """Charge le modèle, le preprocessor et les métadonnées."""
    global model, preprocessor, metadata, imputation_values, optimal_threshold
    
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Modèle chargé: {MODEL_PATH}")
        else:
            print(f"⚠️ Modèle non trouvé: {MODEL_PATH}")
            
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print(f"✅ Preprocessor chargé: {PREPROCESSOR_PATH}")
        else:
            print(f"⚠️ Preprocessor non trouvé: {PREPROCESSOR_PATH}")
            
        if os.path.exists(METADATA_PATH):
            metadata = joblib.load(METADATA_PATH)
            print(f"✅ Métadonnées chargées: {METADATA_PATH}")
            
            # Charger les valeurs d'imputation si disponibles dans les métadonnées
            if isinstance(metadata, dict) and 'imputation_values' in metadata:
                imputation_values = metadata['imputation_values']
                print(f"✅ Valeurs d'imputation chargées")
            else:
                print(f"⚠️ Pas de valeurs d'imputation dans les métadonnées, utilisation de valeurs par défaut")
                imputation_values = None
                
            # Charger le seuil optimal si disponible
            if isinstance(metadata, dict) and 'optimal_threshold' in metadata:
                optimal_threshold = metadata['optimal_threshold']
                print(f"✅ Seuil optimal chargé: {optimal_threshold:.3f}")
            else:
                optimal_threshold = 0.5
                print(f"⚠️ Pas de seuil optimal dans les métadonnées, utilisation de 0.5")
        else:
            print(f"⚠️ Métadonnées non trouvées: {METADATA_PATH}")
            
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")

# Chargement au démarrage
@app.on_event("startup")
async def startup_event():
    load_models()

# ============================================================================
# SCHÉMAS PYDANTIC
# ============================================================================

class EmployeeData(BaseModel):
    """Schéma pour les données d'un employé."""
    # Données générales - Tous les champs sont optionnels
    BusinessTravel: Optional[str] = Field(None, description="Fréquence de voyage (Travel_Rarely, Travel_Frequently, Non-Travel)")
    Department: Optional[str] = Field(None, description="Département (Sales, Research & Development, Human Resources)")
    DistanceFromHome: Optional[int] = Field(None, ge=0, description="Distance domicile-travail (km)")
    Education: Optional[int] = Field(None, ge=1, le=5, description="Niveau d'éducation (1-5)")
    EducationField: Optional[str] = Field(None, description="Domaine d'études")
    JobInvolvement: Optional[int] = Field(None, ge=1, le=4, description="Implication au travail (1-4)")
    JobLevel: Optional[int] = Field(None, ge=1, le=5, description="Niveau hiérarchique (1-5)")
    JobRole: Optional[str] = Field(None, description="Rôle/Poste")
    JobSatisfaction: Optional[int] = Field(None, ge=1, le=4, description="Satisfaction au travail (1-4)")
    MonthlyIncome: Optional[float] = Field(None, ge=0, description="Salaire mensuel")
    NumCompaniesWorked: Optional[int] = Field(None, ge=0, description="Nombre d'entreprises précédentes")
    PercentSalaryHike: Optional[float] = Field(None, ge=0, description="Pourcentage d'augmentation")
    PerformanceRating: Optional[int] = Field(None, ge=1, le=4, description="Note de performance (1-4)")
    StockOptionLevel: Optional[int] = Field(None, ge=0, le=3, description="Niveau de stock options (0-3)")
    TotalWorkingYears: Optional[int] = Field(None, ge=0, description="Années d'expérience totales")
    TrainingTimesLastYear: Optional[int] = Field(None, ge=0, description="Formations suivies l'an dernier")
    WorkLifeBalance: Optional[int] = Field(None, ge=1, le=4, description="Équilibre vie pro/perso (1-4)")
    YearsAtCompany: Optional[int] = Field(None, ge=0, description="Années dans l'entreprise")
    YearsInCurrentRole: Optional[int] = Field(None, ge=0, description="Années dans le poste actuel")
    YearsSinceLastPromotion: Optional[int] = Field(None, ge=0, description="Années depuis dernière promotion")
    YearsWithCurrManager: Optional[int] = Field(None, ge=0, description="Années avec le manager actuel")
    
    # Données de sondage employé
    EnvironmentSatisfaction: Optional[int] = Field(None, ge=1, le=4, description="Satisfaction environnement (1-4)")
    
    # Données temporelles (optionnelles - moyennes calculées)
    Arrive_mean: Optional[float] = Field(None, description="Heure d'arrivée moyenne (ex: 9.5 = 9h30)")
    Worktime_mean: Optional[float] = Field(None, description="Heures de travail moyennes par jour")

class PredictionResponse(BaseModel):
    """Schéma pour la réponse de prédiction."""
    employee_id: Optional[str]
    prediction: str
    probability: float
    risk_level: str
    confidence_level: str
    confidence_message: str
    missing_fields_count: int
    total_fields: int
    risk_factors: List[Dict[str, Any]]

# ============================================================================
# CONSTANTES ET VALEURS PAR DÉFAUT
# ============================================================================

# Colonnes attendues par le modèle (après nettoyage du notebook)
EXPECTED_COLUMNS = [
    'BusinessTravel', 'Department', 'DistanceFromHome', 'Education',
    'EducationField', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobLevel', 'JobRole', 'JobSatisfaction',
    'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
    'PerformanceRating', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'Arrive_mean', 'Worktime_mean'
]

# Colonnes à supprimer (comme dans le notebook)
COLUMNS_TO_DROP = [
    'EmployeeID', 'Gender', 'Over18', 'StandardHours', 
    'EmployeeCount', 'Departure_mean', 'Age', 'Attrition', 'MaritalStatus'
]

# Valeurs par défaut de secours pour l'imputation (si pas dans metadata)
FALLBACK_IMPUTATION_VALUES = {
    # Valeurs numériques - médianes calculées depuis les données d'entraînement
    'DistanceFromHome': 7,
    'Education': 3,
    'EnvironmentSatisfaction': 3,
    'JobInvolvement': 3,
    'JobLevel': 2,
    'JobSatisfaction': 3,
    'MonthlyIncome': 49190,
    'NumCompaniesWorked': 2,
    'PercentSalaryHike': 15,
    'PerformanceRating': 3,
    'StockOptionLevel': 1,
    'TotalWorkingYears': 10,
    'TrainingTimesLastYear': 3,
    'WorkLifeBalance': 3,
    'YearsAtCompany': 7,
    'YearsInCurrentRole': 2,
    'YearsSinceLastPromotion': 1,
    'YearsWithCurrManager': 2,
    'Arrive_mean': 9.0,
    'Worktime_mean': 8.5,
    
    # Valeurs catégorielles - modes
    'BusinessTravel': 'Travel_Rarely',
    'Department': 'Research & Development',
    'EducationField': 'Life Sciences',
    'JobRole': 'Sales Executive'
}

def get_imputation_value(column: str) -> Any:
    """Récupère la valeur d'imputation pour une colonne."""
    if imputation_values and column in imputation_values:
        return imputation_values[column]
    elif column in FALLBACK_IMPUTATION_VALUES:
        return FALLBACK_IMPUTATION_VALUES[column]
    else:
        return 0  # Dernière valeur de secours

# Options pour les champs catégoriels
CATEGORICAL_OPTIONS = {
    'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
    'Department': ['Human Resources', 'Research & Development', 'Sales'],
    'EducationField': ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'],
    'JobRole': [
        'Healthcare Representative', 'Human Resources', 'Laboratory Technician',
        'Manager', 'Manufacturing Director', 'Research Director',
        'Research Scientist', 'Sales Executive', 'Sales Representative'
    ]
}

# Facteurs de risque et recommandations associées
RISK_FACTORS_CONFIG = {
    'JobSatisfaction': {
        'threshold': 2,
        'operator': '<=',
        'message': 'Satisfaction au travail faible',
        'recommendations': [
            'Organiser un entretien individuel pour comprendre les sources d\'insatisfaction',
            'Envisager une redéfinition du poste ou de nouvelles responsabilités',
            'Proposer un mentorat ou un coaching professionnel'
        ]
    },
    'WorkLifeBalance': {
        'threshold': 2,
        'operator': '<=',
        'message': 'Mauvais équilibre vie pro/perso',
        'recommendations': [
            'Proposer des horaires flexibles ou du télétravail',
            'Revoir la charge de travail et les délais',
            'Mettre en place des programmes de bien-être'
        ]
    },
    'EnvironmentSatisfaction': {
        'threshold': 2,
        'operator': '<=',
        'message': 'Insatisfaction avec l\'environnement de travail',
        'recommendations': [
            'Améliorer l\'espace de travail (ergonomie, équipement)',
            'Renforcer la cohésion d\'équipe via des activités',
            'Évaluer les relations avec les collègues et managers'
        ]
    },
    'YearsSinceLastPromotion': {
        'threshold': 3,
        'operator': '>',
        'message': 'Pas de promotion depuis longtemps',
        'recommendations': [
            'Discuter des perspectives d\'évolution de carrière',
            'Proposer des formations pour développer de nouvelles compétences',
            'Envisager une promotion ou une augmentation de responsabilités'
        ]
    },
    'MonthlyIncome': {
        'threshold': 3000,
        'operator': '<',
        'message': 'Salaire potentiellement en dessous du marché',
        'recommendations': [
            'Réaliser un benchmark salarial du marché',
            'Envisager une révision salariale',
            'Proposer des avantages non-monétaires (formation, flexibilité)'
        ]
    },
    'DistanceFromHome': {
        'threshold': 20,
        'operator': '>',
        'message': 'Distance domicile-travail importante',
        'recommendations': [
            'Proposer du télétravail partiel',
            'Ajuster les horaires pour éviter les heures de pointe',
            'Évaluer les options de relocalisation ou indemnités transport'
        ]
    },
    'TrainingTimesLastYear': {
        'threshold': 1,
        'operator': '<',
        'message': 'Peu de formations suivies',
        'recommendations': [
            'Proposer un plan de développement personnalisé',
            'Identifier les formations pertinentes pour le poste',
            'Encourager la participation à des conférences ou workshops'
        ]
    },
    'StockOptionLevel': {
        'threshold': 0,
        'operator': '==',
        'message': 'Aucune participation au capital',
        'recommendations': [
            'Évaluer l\'éligibilité aux stock options',
            'Proposer des alternatives (bonus, intéressement)',
            'Communiquer sur les perspectives de valorisation'
        ]
    },
    'Worktime_mean': {
        'threshold': 10,
        'operator': '>',
        'message': 'Heures de travail excessives',
        'recommendations': [
            'Analyser la charge de travail et redistribuer si nécessaire',
            'Recruter des ressources supplémentaires',
            'Mettre en place des limites horaires claires'
        ]
    }
}

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def preprocess_employee_data(data: dict) -> tuple:
    """
    Prétraite les données d'un employé pour la prédiction.
    Applique les mêmes transformations que dans le notebook.
    Retourne un tuple (DataFrame, missing_count, total_count)
    """
    # Compter les valeurs manquantes
    total_fields = len(EXPECTED_COLUMNS)
    missing_count = sum(1 for col in EXPECTED_COLUMNS if col not in data or data[col] is None or data[col] == '')
    
    # Créer DataFrame
    df = pd.DataFrame([data])
    
    # Supprimer les colonnes non nécessaires
    for col in COLUMNS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Imputer les valeurs manquantes pour toutes les colonnes attendues
    for col in EXPECTED_COLUMNS:
        if col not in df.columns or pd.isna(df[col].iloc[0]) or df[col].iloc[0] == '' or df[col].iloc[0] is None:
            imputation_value = get_imputation_value(col)
            df[col] = imputation_value
    
    # S'assurer que toutes les colonnes attendues sont présentes
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = get_imputation_value(col)
    
    # Réordonner les colonnes
    df = df[[col for col in EXPECTED_COLUMNS if col in df.columns]]
    
    return df, missing_count, total_fields

def analyze_risk_factors(data: dict) -> tuple:
    """
    Analyse les facteurs de risque d'un employé.
    Retourne une liste de facteurs de risque et de recommandations.
    """
    risk_factors = []
    recommendations = set()
    
    for field, config in RISK_FACTORS_CONFIG.items():
        if field not in data or data[field] is None:
            continue
            
        value = data[field]
        threshold = config['threshold']
        operator = config['operator']
        
        # Skip if value is not comparable (not a number/string)
        if not isinstance(value, (int, float, str)):
            continue
        
        is_risk = False
        if operator == '<=' and value <= threshold:
            is_risk = True
        elif operator == '<' and value < threshold:
            is_risk = True
        elif operator == '>' and value > threshold:
            is_risk = True
        elif operator == '>=' and value >= threshold:
            is_risk = True
        elif operator == '==' and value == threshold:
            is_risk = True
            
        if is_risk:
            risk_factors.append({
                'factor': field,
                'value': value,
                'threshold': threshold,
                'message': config['message']
            })
            recommendations.update(config['recommendations'])
    
    return risk_factors, list(recommendations)

def get_risk_level(probability: float) -> str:
    """Détermine le niveau de risque basé sur la probabilité."""
    if probability >= 0.7:
        return 'CRITIQUE'
    elif probability >= 0.5:
        return 'ÉLEVÉ'
    elif probability >= 0.3:
        return 'MODÉRÉ'
    else:
        return 'FAIBLE'

def get_confidence_level(missing_count: int, total_count: int) -> tuple:
    """
    Détermine le niveau de confiance basé sur le pourcentage de valeurs manquantes.
    Retourne (niveau, message)
    """
    if total_count == 0:
        return ('Inconnu', 'Impossible de calculer la confiance')
    
    missing_percentage = (missing_count / total_count) * 100
    
    if missing_percentage <= 10:
        return ('Haute confiance', 'La prédiction est très fiable (peu de valeurs manquantes)')
    elif missing_percentage <= 30:
        return ('Confiance moyenne', 'La prédiction est fiable mais quelques valeurs ont été estimées')
    else:
        return ('Faible confiance', f'Attention : {missing_percentage:.0f}% des champs sont manquants. La prédiction peut être moins précise.')

def predict_single(data: dict) -> dict:
    """Effectue une prédiction pour un seul employé."""
    if model is None or preprocessor is None:
        return {
            'error': 'Modèle non chargé. Vérifiez les chemins des fichiers.',
            'prediction': 'ERREUR',
            'probability': 0.0,
            'risk_level': 'INCONNU',
            'confidence_level': 'Inconnu',
            'confidence_message': 'Modèle non chargé',
            'missing_fields_count': 0,
            'total_fields': 0,
            'risk_factors': []
        }
    
    try:
        # Prétraiter les données et compter les valeurs manquantes
        df, missing_count, total_count = preprocess_employee_data(data)
        
        # Calculer le niveau de confiance
        confidence_level, confidence_message = get_confidence_level(missing_count, total_count)
        
        # Transformer avec le preprocessor
        X_processed = preprocessor.transform(df)
        
        # Probabilité (si disponible)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_processed)[0][1]
            # Utiliser le seuil optimal pour la prédiction
            prediction = 1 if proba >= optimal_threshold else 0
        else:
            # Fallback si pas de predict_proba
            prediction = model.predict(X_processed)[0]
            proba = float(prediction)
        
        # Analyser les facteurs de risque
        risk_factors, _ = analyze_risk_factors(data)
        
        return {
            'prediction': 'DÉPART PROBABLE' if prediction == 1 else 'RESTE PROBABLE',
            'probability': round(float(proba) * 100, 1),
            'risk_level': get_risk_level(proba),
            'confidence_level': confidence_level,
            'confidence_message': confidence_message,
            'missing_fields_count': missing_count,
            'total_fields': total_count,
            'risk_factors': risk_factors
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'prediction': 'ERREUR',
            'probability': 0.0,
            'risk_level': 'ERREUR',
            'confidence_level': 'Inconnu',
            'confidence_message': 'Erreur lors du calcul',
            'missing_fields_count': 0,
            'total_fields': 0,
            'risk_factors': []
        }

# ============================================================================
# ENDPOINTS API
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Page d'accueil avec l'interface RH."""
    model_loaded = model is not None and preprocessor is not None
    model_info = metadata if metadata else {}
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_loaded": model_loaded,
        "model_info": model_info,
        "categorical_options": CATEGORICAL_OPTIONS
    })

@app.get("/api/status")
async def get_status():
    """Vérifie le statut du modèle."""
    return {
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "metadata_loaded": metadata is not None,
        "model_info": metadata if metadata else {}
    }

@app.post("/api/predict/single")
async def predict_employee(employee: EmployeeData):
    """Prédit l'attrition pour un seul employé."""
    result = predict_single(employee.dict())
    return result

@app.post("/api/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Prédit l'attrition pour un fichier CSV d'employés."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Le fichier doit être un CSV")
    
    try:
        # Lire le CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        results = []
        for idx, row in df.iterrows():
            data = row.to_dict()
            result = predict_single(data)
            result['employee_id'] = data.get('EmployeeID', f'Employé_{idx+1}')
            result['row_data'] = {k: v for k, v in data.items() if pd.notna(v)}
            results.append(result)
        
        # Statistiques globales
        predictions = [r['prediction'] for r in results]
        probabilities = [r['probability'] for r in results]
        
        stats = {
            'total': len(results),
            'at_risk': sum(1 for p in predictions if p == 'DÉPART PROBABLE'),
            'safe': sum(1 for p in predictions if p == 'RESTE PROBABLE'),
            'avg_probability': round(np.mean(probabilities), 1),
            'high_risk_count': sum(1 for r in results if r['risk_level'] in ['CRITIQUE', 'ÉLEVÉ'])
        }
        
        # Distribution par département si disponible
        dept_stats = {}
        if 'Department' in df.columns:
            for dept in df['Department'].unique():
                dept_results = [r for i, r in enumerate(results) 
                               if df.iloc[i].get('Department') == dept]
                if dept_results:
                    dept_stats[dept] = {
                        'total': len(dept_results),
                        'at_risk': sum(1 for r in dept_results if r['prediction'] == 'DÉPART PROBABLE'),
                        'avg_probability': round(np.mean([r['probability'] for r in dept_results]), 1)
                    }
        
        return {
            'results': results,
            'statistics': stats,
            'department_stats': dept_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")

@app.get("/api/options")
async def get_options():
    """Retourne les options pour les champs catégoriels."""
    return {
        "categorical_options": CATEGORICAL_OPTIONS,
        "expected_columns": EXPECTED_COLUMNS
    }

@app.get("/api/sample-data")
async def get_sample_data():
    """Retourne un exemple de données pour test."""
    return {
        "BusinessTravel": "Travel_Rarely",
        "Department": "Research & Development",
        "DistanceFromHome": 10,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 3,
        "MonthlyIncome": 5000,
        "NumCompaniesWorked": 2,
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3,
        "Arrive_mean": 9.0,
        "Worktime_mean": 8.5
    }

@app.post("/api/reload-model")
async def reload_model():
    """Recharge le modèle depuis les fichiers."""
    load_models()
    return {
        "status": "reloaded",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
