import unittest
from pathlib import Path
import pandas as pd
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())
from src.data_preprocessing import preprocess


class DataPreprocessingTest(unittest.TestCase):
    # test de regresie, se asigura ca la orice modificare a functiei
    #  de proprocess, datele rezultate raman la fel
    def test_preprocess(self):
        test_data_root_path = Path(__file__).parent / "test_data"
        input_file = test_data_root_path / "input" / "airbnb.csv.zip"
        output_file = test_data_root_path / "out_test" / "df_preprocessed_candiate.csv"
        output_file.parent.mkdir(exist_ok=True)
        reference_file = test_data_root_path / "reference" / "df_processed.csv"
        preprocess(input_file, output_file)

        ref_df = pd.read_csv(reference_file)
        out_test_df = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(ref_df, out_test_df)


if __name__ == "__main__":
    unittest.main()
