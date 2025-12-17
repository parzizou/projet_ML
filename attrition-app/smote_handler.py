"""
🔄 SMOTE Handler Module
=====================
Module pour gérer l'application de SMOTE (Synthetic Minority Over-sampling Technique)
pour équilibrer les classes dans les données d'entraînement.
"""

import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

# Configuration du logger
logger = logging.getLogger(__name__)


class SMOTEConfig:
    """Configuration pour SMOTE."""
    
    def __init__(
        self,
        sampling_strategy: str = 'auto',
        k_neighbors: int = 5,
        random_state: int = 42
    ):
        """
        Initialise la configuration SMOTE.
        
        Args:
            sampling_strategy: Stratégie de rééchantillonnage
                - 'auto': équilibre automatiquement la classe minoritaire
                - float: ratio spécifique (ex: 0.5 pour avoir 50% de la majorité)
                - dict: ratios personnalisés pour chaque classe
            k_neighbors: Nombre de voisins pour générer des échantillons synthétiques
            random_state: Seed pour la reproductibilité
        """
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire."""
        return {
            'sampling_strategy': self.sampling_strategy,
            'k_neighbors': self.k_neighbors,
            'random_state': self.random_state
        }


class SMOTEHandler:
    """Gestionnaire pour l'application de SMOTE."""
    
    def __init__(self, config: Optional[SMOTEConfig] = None):
        """
        Initialise le gestionnaire SMOTE.
        
        Args:
            config: Configuration SMOTE. Si None, utilise la configuration par défaut.
        """
        self.config = config or SMOTEConfig()
        self.smote = None
        self._class_distribution_before = None
        self._class_distribution_after = None
    
    def fit_resample(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applique SMOTE aux données d'entraînement.
        
        Args:
            X_train: Features d'entraînement (array-like ou sparse matrix)
            y_train: Labels d'entraînement
            
        Returns:
            Tuple (X_resampled, y_resampled): Données rééchantillonnées
        """
        # Sauvegarder la distribution avant SMOTE
        self._class_distribution_before = Counter(y_train)
        
        logger.info("=" * 60)
        logger.info("Application de SMOTE pour rééquilibrer les classes")
        logger.info("=" * 60)
        logger.info(f"Configuration SMOTE: {self.config.to_dict()}")
        logger.info(f"Distribution AVANT SMOTE: {dict(self._class_distribution_before)}")
        
        # Créer et appliquer SMOTE
        try:
            self.smote = SMOTE(
                sampling_strategy=self.config.sampling_strategy,
                k_neighbors=self.config.k_neighbors,
                random_state=self.config.random_state
            )
            
            X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
            
            # Sauvegarder la distribution après SMOTE
            self._class_distribution_after = Counter(y_resampled)
            
            logger.info(f"Distribution APRÈS SMOTE: {dict(self._class_distribution_after)}")
            logger.info(f"Nombre d'échantillons AVANT: {len(y_train)}")
            logger.info(f"Nombre d'échantillons APRÈS: {len(y_resampled)}")
            logger.info(f"Échantillons synthétiques créés: {len(y_resampled) - len(y_train)}")
            logger.info("=" * 60)
            
            return X_resampled, y_resampled
            
        except ValueError as e:
            logger.error(f"Erreur lors de l'application de SMOTE: {e}")
            logger.warning("Retour aux données originales sans SMOTE")
            return X_train, y_train
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de l'application de SMOTE.
        
        Returns:
            Dictionnaire contenant les statistiques avant/après
        """
        if self._class_distribution_before is None:
            return {
                'applied': False,
                'message': 'SMOTE n\'a pas encore été appliqué'
            }
        
        return {
            'applied': True,
            'before': dict(self._class_distribution_before),
            'after': dict(self._class_distribution_after),
            'samples_before': sum(self._class_distribution_before.values()),
            'samples_after': sum(self._class_distribution_after.values()),
            'synthetic_samples': (
                sum(self._class_distribution_after.values()) - 
                sum(self._class_distribution_before.values())
            ),
            'config': self.config.to_dict()
        }


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: str = 'auto',
    k_neighbors: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fonction utilitaire pour appliquer SMOTE rapidement.
    
    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        sampling_strategy: Stratégie de rééchantillonnage
        k_neighbors: Nombre de voisins
        random_state: Seed pour la reproductibilité
        
    Returns:
        Tuple (X_resampled, y_resampled, statistics)
    """
    config = SMOTEConfig(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    
    handler = SMOTEHandler(config)
    X_resampled, y_resampled = handler.fit_resample(X_train, y_train)
    statistics = handler.get_statistics()
    
    return X_resampled, y_resampled, statistics
