# 🎯 Application de Prédiction d'Attrition RH

Application web FastAPI pour prédire l'attrition des employés et aider les RH à retenir les talents.

![Interface](https://img.shields.io/badge/Interface-Web-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-red)

## 📋 Fonctionnalités

- ✅ **Prédiction individuelle** : Saisie manuelle des données d'un employé
- ✅ **Import CSV** : Analyse en masse de plusieurs employés
- ✅ **Tableau de bord** : Visualisations et statistiques par département
- ✅ **Facteurs de risque** : Identification automatique des points faibles
- ✅ **Recommandations** : Conseils personnalisés pour améliorer la rétention

## 🚀 Installation

### 1. Cloner le projet

```bash
cd attrition-app
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Placer vos fichiers de modèle

Copiez vos fichiers `.joblib` générés par le notebook dans le dossier `models/` :

```
models/
├── attrition_model.joblib        # Modèle ML entraîné
├── attrition_preprocessor.joblib # Pipeline de preprocessing
└── attrition_metadata.joblib     # Métadonnées (optionnel)
```

**Important** : Ces fichiers sont générés à la fin de votre notebook avec :
```python
joblib.dump(best_model_final, 'attrition_model.joblib')
joblib.dump(preprocessor, 'attrition_preprocessor.joblib')
joblib.dump(metadata, 'attrition_metadata.joblib')
```

### 5. Lancer l'application

```bash
python app.py
# ou
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 6. Accéder à l'interface

Ouvrez votre navigateur à l'adresse : **http://localhost:8000**

## ⚙️ Configuration

Vous pouvez modifier les chemins des fichiers de modèle via variables d'environnement :

```bash
export MODEL_PATH="chemin/vers/votre/modele.joblib"
export PREPROCESSOR_PATH="chemin/vers/votre/preprocessor.joblib"
export METADATA_PATH="chemin/vers/votre/metadata.joblib"
```

Ou directement dans `app.py` :

```python
MODEL_PATH = "models/attrition_model.joblib"
PREPROCESSOR_PATH = "models/attrition_preprocessor.joblib"
METADATA_PATH = "models/attrition_metadata.joblib"
```

## 📊 Format du CSV

Pour l'import CSV, utilisez le format suivant (téléchargeable depuis l'interface) :

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| EmployeeID | string | Identifiant unique | "EMP001" |
| BusinessTravel | string | Fréquence de voyage | "Travel_Rarely" |
| Department | string | Département | "Research & Development" |
| DistanceFromHome | int | Distance domicile (km) | 10 |
| Education | int | Niveau d'éducation (1-5) | 3 |
| EducationField | string | Domaine d'études | "Life Sciences" |
| EnvironmentSatisfaction | int | Satisfaction env. (1-4) | 3 |
| JobInvolvement | int | Implication (1-4) | 3 |
| JobLevel | int | Niveau hiérarchique (1-5) | 2 |
| JobRole | string | Poste | "Research Scientist" |
| JobSatisfaction | int | Satisfaction travail (1-4) | 3 |
| MaritalStatus | string | Statut marital | "Married" |
| MonthlyIncome | float | Salaire mensuel | 5000 |
| NumCompaniesWorked | int | Entreprises précédentes | 2 |
| PercentSalaryHike | float | % augmentation | 15 |
| PerformanceRating | int | Performance (1-4) | 3 |
| StockOptionLevel | int | Stock options (0-3) | 1 |
| TotalWorkingYears | int | Expérience totale | 8 |
| TrainingTimesLastYear | int | Formations | 3 |
| WorkLifeBalance | int | Équilibre (1-4) | 3 |
| YearsAtCompany | int | Ancienneté | 5 |
| YearsInCurrentRole | int | Années dans le poste | 3 |
| YearsSinceLastPromotion | int | Années depuis promo | 1 |
| YearsWithCurrManager | int | Années avec manager | 3 |
| Arrive_mean | float | Heure arrivée moyenne | 9.0 |
| Worktime_mean | float | Heures travail/jour | 8.5 |

### Valeurs acceptées pour les champs catégoriels

- **BusinessTravel** : `Non-Travel`, `Travel_Rarely`, `Travel_Frequently`
- **Department** : `Human Resources`, `Research & Development`, `Sales`
- **EducationField** : `Human Resources`, `Life Sciences`, `Marketing`, `Medical`, `Other`, `Technical Degree`
- **JobRole** : `Healthcare Representative`, `Human Resources`, `Laboratory Technician`, `Manager`, `Manufacturing Director`, `Research Director`, `Research Scientist`, `Sales Executive`, `Sales Representative`
- **MaritalStatus** : `Divorced`, `Married`, `Single`

## 🔌 API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Interface web principale |
| `/api/status` | GET | Statut du modèle |
| `/api/predict/single` | POST | Prédiction pour un employé |
| `/api/predict/csv` | POST | Prédiction pour un fichier CSV |
| `/api/options` | GET | Options des champs catégoriels |
| `/api/sample-data` | GET | Données d'exemple |
| `/api/reload-model` | POST | Recharger le modèle |

### Exemple d'appel API

```bash
curl -X POST "http://localhost:8000/api/predict/single" \
     -H "Content-Type: application/json" \
     -d '{
       "BusinessTravel": "Travel_Rarely",
       "Department": "Research & Development",
       "DistanceFromHome": 10,
       "Education": 3,
       "EducationField": "Life Sciences",
       "EnvironmentSatisfaction": 3,
       "JobInvolvement": 3,
       "JobLevel": 2,
       "JobRole": "Research Scientist",
       "JobSatisfaction": 2,
       "MaritalStatus": "Married",
       "MonthlyIncome": 3000,
       "NumCompaniesWorked": 2,
       "PercentSalaryHike": 11,
       "PerformanceRating": 3,
       "StockOptionLevel": 0,
       "TotalWorkingYears": 8,
       "TrainingTimesLastYear": 0,
       "WorkLifeBalance": 2,
       "YearsAtCompany": 5,
       "YearsInCurrentRole": 3,
       "YearsSinceLastPromotion": 4,
       "YearsWithCurrManager": 3,
       "Arrive_mean": 9.0,
       "Worktime_mean": 10.5
     }'
```

### Réponse

```json
{
  "prediction": "DÉPART PROBABLE",
  "probability": 72.5,
  "risk_level": "CRITIQUE",
  "risk_factors": [
    {
      "factor": "JobSatisfaction",
      "value": 2,
      "threshold": 2,
      "message": "Satisfaction au travail faible"
    },
    {
      "factor": "WorkLifeBalance",
      "value": 2,
      "threshold": 2,
      "message": "Mauvais équilibre vie pro/perso"
    }
  ],
  "recommendations": [
    "Organiser un entretien individuel pour comprendre les sources d'insatisfaction",
    "Proposer des horaires flexibles ou du télétravail"
  ]
}
```

## 🛠️ Structure du projet

```
attrition-app/
├── app.py                 # Application FastAPI principale
├── requirements.txt       # Dépendances Python
├── README.md             # Ce fichier
├── models/               # Fichiers de modèle ML
│   ├── attrition_model.joblib
│   ├── attrition_preprocessor.joblib
│   └── attrition_metadata.joblib
├── templates/            # Templates HTML
│   └── index.html
└── static/              # Fichiers statiques (si nécessaire)
```

## 📈 Interprétation des résultats

### Niveaux de risque

| Niveau | Probabilité | Action recommandée |
|--------|-------------|-------------------|
| 🟢 FAIBLE | < 30% | Surveillance normale |
| 🟡 MODÉRÉ | 30-50% | Attention particulière |
| 🟠 ÉLEVÉ | 50-70% | Action préventive |
| 🔴 CRITIQUE | > 70% | Intervention urgente |

### Facteurs de risque analysés

L'application identifie automatiquement les points faibles :

- **Satisfaction** : Travail, environnement, équilibre vie pro/perso
- **Carrière** : Temps depuis dernière promotion, niveau de rémunération
- **Conditions** : Distance domicile, heures de travail excessives
- **Développement** : Manque de formations, pas de stock options

## 🤝 Contribution

Pour toute amélioration ou bug, n'hésitez pas à ouvrir une issue ou une pull request.

## 📝 Licence

Ce projet est fourni à des fins éducatives et professionnelles.

---

Développé avec ❤️ pour les équipes RH
