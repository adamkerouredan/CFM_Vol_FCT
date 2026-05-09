"""
neutralizer.py
--------------
Responsabilité unique : retirer la composante systématique cross-sectionnelle
des features et de la target.

Pour chaque date t, on soustrait la moyenne cross-sectionnelle (sur les stocks)
afin d'obtenir la composante idiosyncratique du signal.

Référence : Paleologo (2024), Chapitre 6 -- Cross-Sectional Regression.
Le concept est analogue à la procédure de "factor model demeaning" :
on retire la composante commune pour ne garder que la composante
spécifique au stock.
"""

import numpy as np
import pandas as pd


class Neutralizer:
    """
    Neutralise les features et la target par la moyenne cross-sectionnelle
    par date.

    Pour chaque date t :
        f_neutral_{i,t} = f_{i,t} - mean_j( f_{j,t} )

    Parameters
    ----------
    date_column : str
        Nom de la colonne contenant l'identifiant de date.
    """

    def __init__(self, date_column: str = "date") -> None:
        self.date_column = date_column

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def neutralize_features(
        self,
        features: pd.DataFrame,
        meta: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Neutralise toutes les colonnes numériques d'un DataFrame de features.

        Parameters
        ----------
        features : pd.DataFrame
            Features à neutraliser. Doit contenir une colonne ID.
        meta : pd.DataFrame
            DataFrame contenant ID et date_column. Permet de récupérer
            la date associée à chaque ligne.

        Returns
        -------
        pd.DataFrame
            Features neutralisées (mêmes colonnes que l'entrée).
        """
        merged = self._merge_with_date(features, meta)
        feature_cols = self._get_feature_columns(features)

        neutralized = merged.copy()
        date_groups = merged.groupby(self.date_column)

        for col in feature_cols:
            cross_section_mean = date_groups[col].transform("mean")
            neutralized[col] = merged[col] - cross_section_mean

        return neutralized[features.columns].reset_index(drop=True)

    def neutralize_target(
        self,
        y: pd.Series,
        meta: pd.DataFrame,
    ) -> pd.Series:
        """
        Neutralise une série cible par la moyenne cross-sectionnelle.

        Parameters
        ----------
        y : pd.Series
            Target (typiquement log(TARGET)). L'index doit être aligné
            avec meta.
        meta : pd.DataFrame
            Contient la colonne date_column.

        Returns
        -------
        pd.Series
            Target neutralisée.
        """
        df = pd.DataFrame({
            "y":               y.values,
            self.date_column:  meta[self.date_column].values,
        })
        cross_section_mean = df.groupby(self.date_column)["y"].transform("mean")
        return pd.Series(df["y"].values - cross_section_mean.values)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _merge_with_date(
        self,
        features: pd.DataFrame,
        meta: pd.DataFrame,
    ) -> pd.DataFrame:
        """Joint les features avec la colonne date."""
        merged = features.merge(
            meta[["ID", self.date_column]],
            on="ID",
            how="left",
        )
        return merged

    def _get_feature_columns(self, features: pd.DataFrame) -> list[str]:
        """Retourne les colonnes numériques (exclut ID)."""
        return [col for col in features.columns if col != "ID"]