import pandas as pd

def detect_anomalies(sentiment_data, financial_data):
    merged_data = pd.merge(sentiment_data, financial_data, on="timestamp", how="inner")
    #threshold
    merged_data["price_change"] = merged_data["price"].pct_change()
    anomalies = merged_data[merged_data["price_change"].abs() > 0.05]  # Example threshold
    return anomalies

if __name__ == "__main__":
    sentiment_data = pd.read_csv("../data/cleaned_text_data.csv")
    financial_data = pd.read_csv("../data/cleaned_financial_data.csv")

    anomalies = detect_anomalies(sentiment_data, financial_data)
    anomalies.to_csv("../data/anomalies.csv", index=False)
    print("Anomalies detected and saved!")
