# The task of this file is to get the data out of the 4 input databases (excel master file, cavity sensor csv, ddm data and UT test data) and orchestrate them to generate a useful input for ML training

def get_csv_cavity_data(input_path):
    '''This function reads all cavity sensor CSV files from the given input path and returns a list of dataframes.'''
    import pandas as pd
    from pathlib import Path

    input_path = Path(input_path)
    csv_files = list(input_path.glob("**/*.csv"))
    dataframes = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source_file'] = csv_file.name
        df['source_rel_path'] = str(csv_file.relative_to(input_path))
        df['trial_folder'] = csv_file.parent.parent.name
        dataframes.append(df)

    return dataframes

def get_ddm_machine_data(input_path):
    '''This function reads all DDM machine data XML files from the given input path and returns a list of dataframes.'''
    import pandas as pd
    from pathlib import Path
    from data_processing.read_xml_file import load_xml_data, xml_to_long_dataframe

    input_path = Path(input_path)
    xml_files = list(input_path.glob("**/*.xml*"))
    dataframes = []

    for xml_file in xml_files:
        xml_data = load_xml_data(xml_file)
        long_df = xml_to_long_dataframe(xml_data)
        long_df['source_file'] = xml_file.name
        long_df['source_rel_path'] = str(xml_file.relative_to(input_path))
        long_df['trial_folder'] = _trial_folder(xml_file)
        dataframes.append(long_df)

    return dataframes

def get_ut_test_data(input_path):
    '''This function reads all UT test data files from the given input path and returns a list of dataframes.'''
    import pandas as pd
    from pathlib import Path

    input_path = Path(input_path)
    ut_files = list(input_path.glob("**/*.ut*"))
    dataframes = []

    for ut_file in ut_files:
        df = pd.read_csv(ut_file)  # Assuming UT test data is in CSV format; adjust if different
        df['source_file'] = ut_file.name
        df['source_rel_path'] = str(ut_file.relative_to(input_path))
        df['trial_folder'] = ut_file.parent.parent.name
        dataframes.append(df)

    return dataframes

def get_excel_master_data(input_path):
    '''This function reads the Excel master file from the given input path and returns a dataframe.'''
    import pandas as pd
    from pathlib import Path

    input_path = Path(input_path)
    excel_files = list(input_path.glob("**/*.xlsx"))

    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in: {input_path}")

    # Assuming there's only one master Excel file; adjust if there are multiple
    master_file = excel_files[0]
    df = pd.read_excel(master_file)
    df['source_file'] = master_file.name
    df['source_rel_path'] = str(master_file.relative_to(input_path))
    df['trial_folder'] = master_file.parent.parent.name

    return df


def merge_dataframes(dataframes):
    '''This function merges a list of dataframes into a single dataframe.'''
    import pandas as pd

    # Master key for all of this will be the PART_LABEL
    # We have to make sure, that for each dataframe, the PART_LABEL is present and correctly formatted
    for df in dataframes:   
        if 'PART_LABEL' not in df.columns:
            raise KeyError("PART_LABEL column is missing in one of the dataframes.")
        # Ensure PART_LABEL is a string and strip whitespace
        df['PART_LABEL'] = df['PART_LABEL'].astype(str).str.strip()
    
    # In the final 


    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df




if __name__ == "__main__":
    # First step is to get all data from the 4 datasources
    cavity_data = get_csv_cavity_data("data/cavity_sensors")
    ddm_data = get_ddm_machine_data("data/ddd")
    ut_data = get_ut_test_data("data/ut_tests")
    excel_data = get_excel_master_data("data/excel_master")

    #Now we want to merge them to a single dataframe

    INCLUDE_DDM_DATA = False
    INCLUDE_CAVITY_SENSOR_DATA = False


    merged_data = merge_dataframes([cavity_data, ddm_data, ut_data, excel_data])