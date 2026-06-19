import gzip
import xml.etree.ElementTree as ET
import pandas as pd

def convert_xml_gz_to_csv(in_file_path: str, out_file_path: str = None, pattern: str = None) -> None:
    # Open and parse the .gz file
    with gzip.open(in_file_path, 'rb') as f:
        tree = ET.parse(f)
        root = tree.getroot()

    curves = root.findall(".//curveObject")
    
    if not curves:
        print("No data curves found.")
        return
    
    # List to store individual dataframes for each curve
    dataframes = []

    for curve in curves:
        short_name = curve.find('shortText').text or "unknown"
        long_name = curve.find('longText').text or "unknown"
        unit = curve.find('unitText').text or ""

        # Apply filter
        if pattern is not None and pattern.lower() not in long_name.lower() and pattern.lower() not in short_name.lower():
            continue
        
        times = []
        values = []

        samples = curve.findall(".//aSample")
        for sample in samples:
            times.append(int(sample.get('timeUs')))
            values.append(float(sample.get('CAy')))

        if not times:
            continue

        # Create a dataframe for this specific curve
        # We use the long name and unit as the column header
        col_name = f"{long_name} ({short_name} in {unit})".strip()
        df_curve = pd.DataFrame({
            'timeUs': times,
            col_name: values
        })
        
        # Set timeUs as index for easier merging
        df_curve.set_index('timeUs', inplace=True)
        dataframes.append(df_curve)

    if not dataframes:
        print("No curves matched the pattern or contained data.")
        return

    # Merge all dataframes on the timeUs index (outer join)
    # This keeps every unique timestamp from every curve
    final_df = pd.concat(dataframes, axis=1, join='outer').sort_index()

    # Create a global 'time in seconds' column starting from the first recorded sample
    t0 = final_df.index.min()
    final_df.insert(0, 'time [s]', (final_df.index - t0) / 1_000_000)
    final_df.set_index('time [s]', inplace=True)

    # Reset index to make timeUs a normal column again
    final_df.reset_index(inplace=True)

    # Save to CSV
    if out_file_path is None:
        out_file_path = in_file_path.replace(".xml.gz", ".csv")
    final_df.to_csv(out_file_path, index=False)
    print(f"Successfully saved to {out_file_path}")

# Usage Example:
convert_xml_gz_to_csv(r'C:\Users\KoltzenburgNils\OneDrive - inspire AG\Diss\93_Buehler\Versuche_Apr26/ddm_y_FliesslaengerformBuehler_2097_ok.xml.gz', out_file_path=r"C:\Users\KoltzenburgNils\OneDrive - inspire AG\Diss\93_Buehler\Versuche_Apr26/plunger_velocity_2097_machine_data.csv", pattern="v I ")