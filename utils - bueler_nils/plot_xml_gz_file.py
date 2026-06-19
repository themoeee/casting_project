import gzip
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def plot_compressed_xml(file_path, pattern: str = None):
    # Open the .gz file directly
    with gzip.open(file_path, 'rb') as f:
        # Parse the XML from the decompressed stream
        tree = ET.parse(f)
        root = tree.getroot()

    # Find all curve objects in the file
    curves = root.findall(".//curveObject")
    
    if not curves:
        print("No data curves found.")
        return

    plt.figure(figsize=(16, 6))

    for curve in curves:
        # Extract metadata for the legend and labels
        short_name = curve.find('shortText').text
        long_name = curve.find('longText').text
        unit = curve.find('unitText').text

        # apply filter for variable
        if pattern is not None and pattern not in long_name and pattern not in short_name:
            continue
        
        times = []
        values = []

        # Find all analog samples (aSample)
        samples = curve.findall(".//aSample")
        for sample in samples:
            # timeUs: timestamp in microseconds
            # CAy: the data value
            times.append(int(sample.get('timeUs')))
            values.append(float(sample.get('CAy')))

        if not times:
            continue

        # Convert microseconds to relative seconds starting from 0
        t0 = times[0]
        time_seconds = [(t - t0) / 1_000_000 for t in times]

        plt.plot(time_seconds, values, label=f"{long_name} ({short_name}), unit: {unit}")

    plt.title("Time Series Data")
    plt.xlabel("Cycle time in s")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Usage:
# plot_compressed_xml(r'C:\Users\KoltzenburgNils\OneDrive - inspire AG\Diss\93_Buehler\Versuche_Sept25/ddd_y_250905132630.xml.gz')
plot_compressed_xml(r'C:\Users\KoltzenburgNils\OneDrive - inspire AG\Diss\93_Buehler\Versuche_Apr26/ddm_y_FliesslaengerformBuehler_2097_ok.xml.gz', pattern=None)