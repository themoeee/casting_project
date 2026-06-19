import pandas as pd
import os
import matplotlib.pyplot as plt

def merge(file1: str, file2: str, outfile: str = None, swap_axes: bool = False) -> None:
    # 1. read files
    df_s = pd.read_csv(file1)
    df_v = pd.read_csv(file2)

    # Capture original column names to restore them later
    # assume col 0 is time and col 1 is the data
    orig_time_name = df_s.columns[0]
    orig_s_name = df_s.columns[1]
    orig_v_name = df_v.columns[1]

    # Rename for clean logic
    df_s = df_s.rename(columns={orig_time_name: 'time', orig_s_name: 's'})
    df_v = df_v.rename(columns={df_v.columns[0]: 'time', orig_v_name: 'v'})

    # 2. Merge the dataframes on 'time'
    # 'outer' join ensures we keep every unique timestamp from both sets
    df_combined = pd.merge(df_s, df_v, on='time', how='outer').sort_values('time')

    # 3. Interpolate the missing values
    # limit_direction='both' handles the start and end of the series
    df_combined[['s', 'v']] = df_combined[['s', 'v']].interpolate(method='linear', limit_direction='both')

    # 4. Filter to the overlapping range (optional but recommended)
    t_min = max(df_s['time'].min(), df_v['time'].min())
    t_max = min(df_s['time'].max(), df_v['time'].max())
    df_final = df_combined[(df_combined['time'] >= t_min) & (df_combined['time'] <= t_max)]

    # plot
    if swap_axes:
        df_final.plot(x="v", y="s", grid=True, xlabel=orig_v_name, ylabel=orig_s_name)
    else:
        df_final.plot(x="s", y="v", grid=True, xlabel=orig_s_name, ylabel=orig_v_name)
    plt.show()

    df_final = df_final.rename(columns={
            'time': orig_time_name,
            's': orig_s_name,
            'v': orig_v_name
        })

    # save
    if outfile is not None:
        if not os.path.isabs(outfile):
            outpath = os.path.join(os.path.dirname(file1), outfile)
        df_final.to_csv(outpath, index=False)
        print(f"Saved merged data to: {outpath}")


if __name__ == "__main__":
    merge(r"C:\Users\KoltzenburgNils\OneDrive - inspire AG\Diss\93_Buehler\Versuche_Apr26/plunger_velocity_2097_machine_data.csv",
          r"C:\Users\KoltzenburgNils\OneDrive - inspire AG\Diss\93_Buehler\Versuche_Apr26/plunger_displacement_2097_machine_data.csv",
          "stroke_velocity_2097_machine_data.csv", swap_axes=True)