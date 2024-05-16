import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mp
from scipy.optimize import curve_fit
import time

# Curve functions
def PS(x, a, b, c):
    return (a - b) * (2 ** -(x / c)) + b
PS.label = r' $y = (a-b)(2^{-(\frac{x}{C})})+b$'

def EC90(x, a, b, c):
    return a + (b - a) / (1 + (c / x))
EC90.label = r'$y = (\frac{(b-a)}{1+(\frac{c}{x})})+a$'

# Corresponding funcline functions for each model
def funcline_PS(popt, y):
    a, b, c = popt
    return (np.log(((b * y) - b) / (a - b)) / -np.log(2)) * c

def funcline_EC90(popt, y):
    a, b, c = popt
    return c * ((a - b) / ((b * y) - b) - 1)

# Define bounds for curve fitting
bounds_PS = ((-np.inf, -np.inf, 0.001), (np.inf, np.inf, 0.15))
bounds_EC90 = ((-np.inf, -np.inf, 0), (np.inf, np.inf, 1))

# Function to process input data
def process_input():
    print("\n(online tool frequence of use is measured without any data colection.\nPress Enter once before your data if you refuse, otherwise ignore)\n\n\nCopy/Paste the data from Compass and then press ENTER twice:\n")
    titles = input()  # Get the titles or an empty row to trigger secret mode

    # Check if the input for titles is empty
    if titles.strip() == "":
        secret_mode = True
        titles = input()  # Get the actual column headers after activating secret mode
    else:
        secret_mode = False

    # Split the titles into column headers
    columns = [x.strip() for x in titles.split('\t')]
    data = {col: [] for col in columns}

    # Process data entries
    peak_count = 0
    while True:
        candidates = input('\nData Peak {}: '.format(peak_count + 1))
        if not candidates:
            break
        peak_count += 1
        values = [x.strip() for x in candidates.split('\t')]

        # Ensure the number of values matches the number of columns
        if len(values) != len(columns):
            print("Warning: Number of values entered does not match number of columns.")
            continue

        for col, val in zip(columns, values):
            try:
                if col in ['Height', 'Area', 'Baseline', 'S/N']:  # Convert to float if necessary
                    data[col].append(float(val.replace(',', '.')))
                else:
                    data[col].append(val)
            except ValueError as e:
                print(f"Error converting data for column {col}: {e}")

    if secret_mode:
        print("usage of online tool not incremented")
    else:
        pass

    return pd.DataFrame(data)


Axisfont = {'color':  'k',
      'weight': 'normal',
      'size': 20,
      }

def curv_fit_plot(func, funcline, bounds, data):
    try:
        dilution = [1 / float(x.split(':')[1]) for x in data['Primary Attr']]
    except:
        print(
            "FYI: since Compass Version 6.2.0 it is possible to show the attribute primary column in the peaks table. This Online-tool accepts both options.")
        dilution = [1 / float(x.split(':')[1]) for x in data['Primary']]

    peak_area = data['Area']
    if data['Name'].isna().all() or data['Name'].eq('').all():
        Fig_Title = data['Sample'].iloc[0]  # Assuming all entries have the same sample name
    else:
        Fig_Title = f"{data['Sample'].iloc[0]} + {data['Name'].iloc[0]}"

    try:
        popt, _ = curve_fit(func, dilution, peak_area, bounds=bounds)
        print('\n',func.__name__)
        dot_lin_50 = funcline(popt, 0.50)  # Calculate dilution for 90% using funcline
        print("a : ", round(popt[0],2))
        print("b : ", round(popt[1],2))
        print("c : 1/" + str(round(1/dot_lin_50,2)))
        dot_lin_90 = funcline(popt, 0.90)  # Calculate dilution for 90% using funcline
        print("\nAb dilution factor to reach 90% of b ≈ ", int(1/dot_lin_90))
    except RuntimeError as e:
        print("\nFit impossible:", e)
        return

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08})
    axs[0].set_title(Fig_Title, fontsize=26, loc='center', pad=20)
    b = popt[1]
    axs[0].plot(dilution, [100 * p / b for p in peak_area], 'ro', label='Datapoints: Peak Area as % of b')
    x_vals = np.linspace(min(dilution), max(dilution), 400)
    y_vals = [100 * v / b for v in func(x_vals, *popt)]
    axs[0].axvline(x=dot_lin_90, color='g', linestyle=':')
    axs[0].plot(x_vals, y_vals, 'r', linewidth=4, alpha=0.4, label=f'Fit: {func.label}')

    #R²
    residuals = peak_area - func(dilution, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((peak_area - np.mean(peak_area)) ** 2)
    R2 = round(1 - (ss_res / ss_tot), 5)
    if len(peak_area) > 3:
        axs[0].plot([], [], ' ', label=f"$R^2$= {R2}")
    else:
        print("\nWarning: R² cannot be determined with 3 datapoints or fewer")
    #R²

    x_vals_90 = np.linspace(dot_lin_90, max(dilution), 400)
    y_vals_90 = [100 * v / b for v in func(x_vals_90, *popt)]
    axs[0].plot(x_vals_90, y_vals_90, color='g')

    axs[0].set_xticklabels([])
    axs[0].set_xticks([])  # This will remove the x-axis ticks
    axs[0].set_ylabel('Antibody Saturation\n$\\regular_{(\%\ of\ Target\ Saturated)}$', fontdict=Axisfont)
    axs[0].set_ylim(0, ((max(peak_area)*100)/b)*1.05)
    axs[0].set_yticks(range(0, 101, 10))
    axs[0].tick_params(axis='y', labelsize=12)
    axs[0].legend(fontsize=14)

    # Second graph (baseline/peak height ratio, as percentage)
    baseline = data['Baseline']
    peak_height = data['Height']
    ratio_baseline_peak = [100 * b / p for b, p in zip(baseline, peak_height)]

    # RED / GREEN color code
    resolution = 0.0015
    xinterp = np.arange(min(dilution), max(dilution), resolution)
    yinterp = np.interp(xinterp, dilution, ratio_baseline_peak)
    norm = mp.colors.Normalize(vmin=5, vmax=20, clip=True)
    cmap = mp.colormaps['RdYlGn_r']
    my_cmap = cmap(norm(yinterp))
    axs[1].bar(xinterp, yinterp, resolution, color=my_cmap, alpha=0.5)
    # RED / GREEN color code

    axs[1].plot(dilution, ratio_baseline_peak, 'bx', label='Baseline/Peak Height (%)')
    axs[1].plot(dilution, ratio_baseline_peak, 'b', linewidth=3, label='_nolegend_')
    axs[1].axvline(x=dot_lin_90, color='g', linestyle=':')
    axs[1].tick_params(axis='y', labelsize=12)
    axs[1].tick_params(axis='x', labelsize=12)
    axs[1].set_xlabel('Dilution of Primary Antibody', fontdict=Axisfont)
    axs[1].set_ylabel('Baseline\n$\\regular_{(\%\ of\ Peak\ Height)}$', fontdict=Axisfont)
    axs[1].legend()

    # Adjust x-axis to show the reciprocal of dilution values
    axs[1].set_xticks(dilution + [dot_lin_90])
    axs[1].set_xticklabels(
        [f"1:{int(1 / d)}" if d != dot_lin_90 else "" for d in dilution] + [f"1:{int(1 / dot_lin_90)}"], rotation=90)

    # Set tick colors
    tick_labels = axs[1].get_xticklabels()
    for i, label in enumerate(tick_labels):
        label.set_color('green' if i == len(tick_labels) - 1 else 'black')

    #fig.tight_layout()
    fig.savefig(f'Graph_{func.__name__}.png', format='png')

# Example usage
data = process_input()
start_time = time.time()  # save the current time
curv_fit_plot(EC90, funcline_EC90, bounds_EC90, data)
curv_fit_plot(PS, funcline_PS, bounds_PS, data)
end_time = time.time()  # save the current time after your code has run
execution_time = end_time - start_time  # calculate the difference
print(f"The script took {execution_time} seconds to complete.")
