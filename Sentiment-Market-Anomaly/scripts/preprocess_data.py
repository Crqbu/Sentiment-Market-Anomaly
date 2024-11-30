import pandas as pd

def preprocess_text_data(file_path):
    #TO ADD ONCE WE HAVE DONE THE SCRAPPING
    data = pd.read_csv(file_path)
    data["cleaned_text"] = data["text"].str.lower()  # Example: Lowercase text
    return data

def preprocess_financial_data(file_path):
    # TO ADD FROM YAHOOO FINANCE
    data = pd.read_csv(file_path)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data.sort_values("timestamp", inplace=True)
    return data

if __name__ == "__main__":
    text_data = preprocess_text_data("../data/text_data.csv")
    financial_data = preprocess_financial_data("../data/financial_data.csv")

    text_data.to_csv("../data/cleaned_text_data.csv", index=False)
    financial_data.to_csv("../data/cleaned_financial_data.csv", index=False)
    print("Preprocessing complete!")
