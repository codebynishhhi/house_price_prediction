import sys
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


class DataEncoding:
    def __init__(self):
        self.ordinal_cols = {
            "Exter Qual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "Exter Cond": ["Po", "Fa", "TA", "Gd", "Ex"],
            "Bsmt Qual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "Bsmt Cond": ["Po", "Fa", "TA", "Gd", "Ex"],
            "Heating QC": ["Po", "Fa", "TA", "Gd", "Ex"],
            "Kitchen Qual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "Fireplace Qu": ["Po", "Fa", "TA", "Gd", "Ex"],
            "Garage Qual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "Garage Cond": ["Po", "Fa", "TA", "Gd", "Ex"],
        }

        self.ordinal_features = list(self.ordinal_cols.keys())
        self.ordinal_categories = list(self.ordinal_cols.values())

    def get_transformer(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Build column transformer for encoding
        """
        try:
            logger.info("Creating encoding transformer")

            categorical_cols = df.select_dtypes(include="object").columns.tolist()

            nominal_cols = [
                col for col in categorical_cols if col not in self.ordinal_features
            ]

            ordinal_encoder = OrdinalEncoder(
                categories=self.ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )

            onehot_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )

            transformer = ColumnTransformer(
                transformers=[
                    ("ordinal", ordinal_encoder, self.ordinal_features),
                    ("nominal", onehot_encoder, nominal_cols),
                ],
                remainder="passthrough"
            )

            logger.info("Encoding transformer created successfully")
            return transformer

        except Exception as e:
            raise CustomException(e, sys)
