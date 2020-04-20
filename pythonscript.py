import numpy as np
import scipy.signal as sci
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def raspi_import(path, channels=5):
    # Load data from .bin file

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
    return sample_period, data


def butter_bandpass(lowcut, highcut, fs, order):
    # Defining sos for butterworth bandpassfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = sci.butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def signal_prosessing(sig, fs):
    # Bandpassfiltering and upsampling signal

    sos = butter_bandpass(250, 2500, fs, 9)
    sig = sci.sosfilt(sos, sig)

    sig = sci.resample(sig, len(sig) * 16)

    return sig


def plot_correlation(c21, c31, c32, d21, d31, d32):
    # Plot correlation and the found delay

    d = 200

    c21 = c21[int((len(c21) / 2)) - d: int((len(c21) / 2)) + d]
    c31 = c31[int((len(c31) / 2)) - d: int((len(c31) / 2)) + d]
    c32 = c32[int((len(c32) / 2)) - d: int((len(c32) / 2)) + d]

    l = range(-d, d)

    plt.plot(l, abs(c21), 'firebrick')
    plt.plot(l, abs(c31), 'darkolivegreen')
    plt.plot(l, abs(c32), 'steelblue')
    plt.axvline(d21, 0, 5, color='darkslategray')
    plt.axvline(d31, 0, 5, color='darkslategray')
    plt.axvline(d32, 0, 5, color='darkslategray')

    plt.show()


def sound_sample(filename):
    # Load data, process signal and returning the calculated angle

    spl, data = raspi_import("data/" + filename)
    fs = 32125

    mic1 = signal_prosessing(data[5:, 2], fs)[1000:]
    mic2 = signal_prosessing(data[5:, 3], fs)[1000:]
    mic3 = signal_prosessing(data[5:, 4], fs)[1000:]

    c21 = (sci.correlate(mic2, mic1))
    c31 = (sci.correlate(mic3, mic1))
    c32 = (sci.correlate(mic3, mic2))

    d21 = np.argmax(c21) - (len(c21) - 1) / 2
    d31 = np.argmax(c31) - (len(c31) - 1) / 2
    d32 = np.argmax(c32) - (len(c32) - 1) / 2

    angle = np.arctan2(np.sqrt(3) * (d21 + d31), (d21 - d31 - 2 * d32)) * (360 / (2 * np.pi))

    if (angle < 0): angle += 360

    # plot_correlation(c21, c31, c32, d21, d31, d32)

    return angle


def plot_data(data):
    # Plot the results

    fig, axs = plt.subplots(4, 1, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})

    sns.distplot(data[0], ax=axs[0], rug=True, kde=False)
    sns.distplot(data[1], ax=axs[1], rug=True, kde=False, fit=norm)
    sns.distplot(data[2], ax=axs[2], rug=True, kde=False)
    sns.distplot(data[3], ax=axs[3], rug=True, kde=False)

    axs[0].axvline(40, color='k', linestyle='--')
    axs[0].set_ylabel('Måleserie 40°')
    axs[0].set_yticks([])

    axs[1].axvline(60, color='k', linestyle='--')
    axs[1].set_ylabel('Måleserie 60°')
    axs[1].set_yticks([])

    axs[2].axvline(240, color='k', linestyle='--')
    axs[2].set_ylabel('Måleserie 240°')
    axs[2].set_yticks([])

    axs[3].axvline(330, color='k', linestyle='--')
    axs[3].set_ylabel('Måleserie 330°')
    axs[3].set_yticks([])
    axs[3].set_xlabel("Vinkel [°]")

    plt.xlim(0, 360)

    for a in axs.reshape(-1):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)

    plt.show()

def plot_measurements(angles):
    # Plot individual measurements

    plt.hlines(1, 1, 360)  # Draw a horizontal line
    plt.xlim(0, 360)
    plt.ylim(0.5, 1.5)

    for m in angles:
        y = np.ones(len(m))
        plt.plot(m, y, '|', ms=50)

    plt.xlabel("Vinkel [°]")
    plt.axis('off')
    plt.show()

def main():
    sns.set_palette("husl")

    # Measurements
    f40deg = ["0_1.bin", "0_2.bin", "0_3.bin", "0_4.bin", "0_5.bin"]

    f60deg = ["60_1.bin", "60_2.bin", "60_3.bin", "60_4.bin", "60_5.bin",
              "60_6.bin", "60_7.bin", "60_8.bin", "60_9.bin", "60_10.bin",
              "60_11.bin", "60_12.bin", "60_13.bin", "60_14.bin", "60_15.bin",
              "60_16.bin", "60_17.bin", "60_18.bin", "60_19.bin", "60_20.bin"]

    f240deg = ["240_1.bin", "240_2.bin", "240_3.bin", "240_4.bin", "240_5.bin"]

    f330deg = ["330_1.bin", "330_2.bin", "330_3.bin", "330_4.bin", "330_5.bin"]

    files = [f40deg, f60deg, f240deg, f330deg]

    data = [[], [], [], []]

    # Calculate angle for each file and add to data-array
    for n in range(4):
        for filename in files[n]:
            d = sound_sample(filename)
            data[n].append(d)

    # Plot result
    #plot_data(data)
    plot_measurements(data)

    # Calculate average and std
    print("40 grader  - Gjennomsnitt = " + str(np.average(data[0])) + " Std. = " + str(np.std(data[0], ddof=1)))
    print("60 grader  - Gjennomsnitt = " + str(np.average(data[1])) + " Std. = " + str(np.std(data[1], ddof=1)))
    print("240 grader - Gjennomsnitt = " + str(np.average(data[2])) + " Std. = " + str(np.std(data[2], ddof=1)))

    # As one measurement returns 13deg, 360deg is added to get the right average and std.
    for i in range(len(data[3])):
        if data[3][i] < 180:
            data[3][i] = data[3][i] + 360

    print("330 grader - Gjennomsnitt = " + str(np.average(data[3])) + " Std. = " + str(np.std(data[3], ddof=1)))

main()
