import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read all the .csv files from a folder and concatenate into a single DataFrame
def read_all_csvs(folder_path):
    all_dfs = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"No CSV files found in {folder_path}")

    return pd.concat(all_dfs, ignore_index=True)


# Preprocess the merged DataFrame to ensure clean and consistent data
def preprocess_final_data(df):

    df.columns = df.columns.str.strip()

    df.rename(columns={
        "date": "Date",
        "ghi": "GHI",
        "pr": "PR"
    }, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values("Date")

    df["GHI"] = df["GHI"].ffill().fillna(0)
    df["PR"] = df["PR"].ffill().fillna(0)

    return df[["Date", "GHI", "PR"]]



def generate_pr_visualization(df, start_date=None, end_date=None,
                              output_image="output_graph.png"):

    df = df.sort_values("Date").copy()

    # Filter by date range if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df["Date"] >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df["Date"] <= end_date]

    if df.empty:
        raise ValueError("No data available in the specified date range.")

    # 30-day moving average
    df["PR_30MA"] = df["PR"].rolling(window=30).mean()

    # Dynamic Budget Calculation 
    start_budget = 73.9
    reduction_rate = 0.008

    df["FY"] = np.where(
        df["Date"].dt.month >= 7,
        df["Date"].dt.year,
        df["Date"].dt.year - 1
    )

    unique_fy = sorted(df["FY"].unique())

    budget_map = {}
    for i, year in enumerate(unique_fy):
        budget_map[year] = start_budget * ((1 - reduction_rate) ** i)

    df["Budget_PR"] = df["FY"].map(budget_map)

    # Points above budget
    df["Above_Budget"] = df["PR"] > df["Budget_PR"]
    above_budget_count = (
        df.groupby("FY")["Above_Budget"]
        .sum()
        .to_dict()
    )

    # Color coding for GHI
    def get_color(ghi):
        if ghi < 2:
            return "navy"
        elif 2 <= ghi < 4:
            return "lightblue"
        elif 4 <= ghi < 6:
            return "orange"
        else:
            return "brown"

    df["Color"] = df["GHI"].apply(get_color)

    
    plt.figure(figsize=(14, 7))

    plt.scatter(df["Date"], df["PR"], c=df["Color"], s=15)
    plt.plot(df["Date"], df["PR_30MA"], color="red", label="30D Moving Avg")
    plt.plot(df["Date"], df["Budget_PR"], color="darkgreen", label="Budget PR")

    # Bottom-right summary
    last_7 = df["PR"].tail(7).mean()
    last_30 = df["PR"].tail(30).mean()
    last_60 = df["PR"].tail(60).mean()

    summary_text = (
        f"Last 7 Days Avg PR: {last_7:.2f}\n"
        f"Last 30 Days Avg PR: {last_30:.2f}\n"
        f"Last 60 Days Avg PR: {last_60:.2f}\n\n"
        f"Points Above Budget:\n"
    )

    for year, count in above_budget_count.items():
        summary_text += f"{year}-{year+1}: {int(count)}\n"

    plt.gca().text(
        0.98, 0.02,
        summary_text,
        transform=plt.gca().transAxes,
        verticalalignment='bottom',
        horizontalalignment='right',
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    # Legend for GHI colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='GHI < 2',
               markerfacecolor='navy', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='2 ≤ GHI < 4',
               markerfacecolor='lightblue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='4 ≤ GHI < 6',
               markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='GHI > 6',
               markerfacecolor='brown', markersize=8)
    ]

    plt.legend(handles=legend_elements, loc="upper left")

    plt.xlabel("Date")
    plt.ylabel("Performance Ratio (PR)")
    plt.title("PR Performance Evolution")
    plt.tight_layout()

    plt.savefig(output_image, dpi=300)
    print(f" Graph saved as {output_image}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate PR Visualization")
    parser.add_argument("--start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    base_dir = os.path.join("data", "data")  

    ghi_df = read_all_csvs(os.path.join(base_dir, "GHI"))
    pr_df = read_all_csvs(os.path.join(base_dir, "PR"))

    merged_df = pd.merge(ghi_df, pr_df, on="Date", how="outer")
    final_df = preprocess_final_data(merged_df)

    final_df.to_csv("merged_output.csv", index=False)

    generate_pr_visualization(
        final_df,
        start_date=args.start_date,
        end_date=args.end_date
    )
