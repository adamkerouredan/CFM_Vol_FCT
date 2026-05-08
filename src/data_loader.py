"""
data_loader.py
--------------
Responsabilité unique : charger et valider les fichiers de données brutes.
Ne fait aucun traitement, aucune transformation.
"""

from pathlib import Path
import pandas as pd


class DataLoader:
    """
    Charge et valide les fichiers bruts du challenge CFM.

    Attributes
    ----------
    data_dir : Path
        Chemin racine du dossier data/.
    separator : str
        Séparateur utilisé dans les fichiers CSV (défaut : ';').
    """

    EXPECTED_COLUMNS_COUNT = 111  # id + date + product_id + 54 vol + 54 ret

    def __init__(self, data_dir: str, separator: str = ";") -> None:
        self.data_dir = Path(data_dir)
        self.separator = separator
        self._validate_data_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_x_train(self) -> pd.DataFrame:
        """Charge le fichier d'entrée d'entraînement."""
        path = self.data_dir / "X_train" / "training_input.csv"
        return self._load_csv(path)

    def load_y_train(self) -> pd.DataFrame:
        """Charge le fichier cible d'entraînement (séparateur virgule)."""
        path = self.data_dir / "Y_train.csv"
        return self._load_csv(path, separator=",")

    def load_x_test(self) -> pd.DataFrame:
        """Charge le fichier d'entrée de test."""
        path = self.data_dir / "X_test" / "testing_input.csv"
        return self._load_csv(path)

    def load_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Charge les trois fichiers en une seule fois.

        Returns
        -------
        x_train, y_train, x_test : tuple de DataFrames
        """
        x_train = self.load_x_train()
        y_train = self.load_y_train()
        x_test  = self.load_x_test()

        self._validate_x(x_train, label="x_train")
        self._validate_x(x_test,  label="x_test")
        self._validate_y(y_train)
        self._validate_alignment(x_train, y_train)

        return x_train, y_train, x_test

    # ------------------------------------------------------------------
    # Private : chargement
    # ------------------------------------------------------------------

    def _load_csv(self, path: Path, separator: str = None) -> pd.DataFrame:
        """Charge un fichier CSV avec vérification d'existence."""
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        sep = separator if separator is not None else self.separator
        return pd.read_csv(path, sep=sep)

    # ------------------------------------------------------------------
    # Private : validation
    # ------------------------------------------------------------------

    def _validate_data_dir(self) -> None:
        """Vérifie que le dossier data/ existe."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dossier data introuvable : {self.data_dir}")

    def _validate_x(self, df: pd.DataFrame, label: str) -> None:
        """Vérifie la structure d'un fichier d'entrée."""
        if df.shape[1] != self.EXPECTED_COLUMNS_COUNT:
            raise ValueError(
                f"{label} : {df.shape[1]} colonnes trouvées, "
                f"{self.EXPECTED_COLUMNS_COUNT} attendues."
            )
        if df.isnull().any().any():
            n_missing = df.isnull().sum().sum()
            print(f"[WARNING] {label} contient {n_missing} valeurs manquantes.")

    def _validate_y(self, df: pd.DataFrame) -> None:
        """Vérifie la structure du fichier cible."""
        expected_columns = {"ID", "TARGET"}
        if not expected_columns.issubset(df.columns):
            raise ValueError(f"y_train : colonnes attendues {expected_columns}.")
        if (df["TARGET"] < 0).any():
            raise ValueError("y_train : des valeurs cibles négatives détectées.")

    def _validate_alignment(
        self, x_train: pd.DataFrame, y_train: pd.DataFrame
    ) -> None:
        """Vérifie que x_train et y_train ont le même nombre de lignes."""
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"Désalignement : x_train={x_train.shape[0]} lignes, "
                f"y_train={y_train.shape[0]} lignes."
            )