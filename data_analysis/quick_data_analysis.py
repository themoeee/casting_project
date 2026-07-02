# The aim of this script is to quickly show which probes performed best and which performed worst
# We will rank between parameter sets, as this is the most interesting comparison for our use case
# The ranking should be based on different criteria, such as the mean performance, the standard deviation of the performance, and the number of tests performed

import pandas as pd
from pathlib import Path

from data_processing.processing.preprocess_ut_tests import DEFAULT_INPUT_CSV, DEFAULT_INPUT_PARQUET



def load_ut_tests_data(
    input_parquet: str | Path = DEFAULT_INPUT_PARQUET,
    input_csv: str | Path = DEFAULT_INPUT_CSV,
) -> pd.DataFrame:
    """Load the processed UT tests data from the specified input files."""

    if Path(input_parquet).exists():
        return pd.read_parquet(input_parquet)
    elif Path(input_csv).exists():
        return pd.read_csv(input_csv)
    else:
        raise FileNotFoundError(
            f"Neither {input_parquet} nor {input_csv} exist. Please provide valid input files."
        )
    
def check_ut_data(processed: pd.DataFrame) -> None:
    """Check if all UT data is present and valid."""
    required_columns = [
        "casting_part_label",
        "parameter_set",
        "sample_position",
        "part_number",
        "performance_metric",  # Replace with the actual performance metric column name
    ]
    missing_columns = [col for col in required_columns if col not in processed.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in processed data: {missing_columns}")  
    

def rank_ut_tests(AVERAGE_BETWEEN_TRIALS_WITH_SAME_PARAMETER_SET: bool, AVERAGE_INSIDE_PARAMETER_SET: bool) -> pd.DataFrame:
    """Rank the UT tests based on the specified criteria."""
    processed_data = load_ut_tests_data()
    check_ut_data(processed_data)

    # Group by parameter set and sample position to calculate mean performance
    if AVERAGE_BETWEEN_TRIALS_WITH_SAME_PARAMETER_SET:
        grouped = processed_data.groupby(["parameter_set", "sample_position"]).agg(
            mean_performance=("performance_metric", "mean"),
            std_performance=("performance_metric", "std"),
            n_trials=("performance_metric", "count")
        ).reset_index()
    else:
        grouped = processed_data.copy()
        grouped["mean_performance"] = grouped["performance_metric"]
        grouped["std_performance"] = 0
        grouped["n_trials"] = 1

    # Further group by parameter set if required
    if AVERAGE_INSIDE_PARAMETER_SET:
        final_grouped = grouped.groupby("parameter_set").agg(
            mean_performance=("mean_performance", "mean"),
            std_performance=("std_performance", "mean"),
            n_trials=("n_trials", "sum")
        ).reset_index()
    else:
        final_grouped = grouped.copy()

    # Rank based on mean performance (higher is better)
    final_grouped["rank"] = final_grouped["mean_performance"].rank(ascending=False)

    return final_grouped.sort_values(by="rank")




if __name__ == "__main__":
    # Load the processed UT tests data
    processed_data = load_ut_tests_data()

    # Check if all required columns are present
    check_ut_data(processed_data)

    AVERAGE_BETWEEN_TRIALS_WITH_SAME_PARAMETER_SET = True  # Set to False if you want to keep all trials separate
    AVERAGE_INSIDE_PARAMETER_SET = True  # Set to False if you want to keep all parameter sets separate

    