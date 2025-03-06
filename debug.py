import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_file = 'runs/training/final/track1.csv'
    
    # Read the CSV without auto-parsing the time column
    df = pd.read_csv(csv_file)
    
    # Convert the "time" column from epoch seconds to datetime objects
    df["time"] = pd.to_datetime(df["time"], unit='s')
    
    # Ensure the DataFrame is sorted by the time column
    df = df.sort_values(by="time")
    
    # Compute the time differences between consecutive timestamps
    df["time_diff"] = df["time"].diff()
    
    # Convert the time differences to seconds
    df["time_diff_seconds"] = df["time_diff"].dt.total_seconds()
    
    # Drop the first row which has NaN for the difference
    time_differences = df["time_diff_seconds"].dropna()
    
    # Plot the distribution of time differences
    plt.figure(figsize=(10, 6))
    plt.hist(time_differences, bins=1000, edgecolor="black")
    plt.title("Distribution of Time Differences Between Sample Points")
    plt.xlabel("Time Difference (seconds)")
    plt.ylabel("Frequency")
    plt.xlim([0,0.04])
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
