# Sentiment-Driven Market Anomaly Detection  

This project explores the relationship between sentiment data (e.g., news, social media) and financial market anomalies such as flash crashes. By integrating high-frequency financial data with sentiment analysis, we aim to detect, analyze, and predict market behaviors influenced by external sentiment signals.  

---

## **Team Members**  
- **Mert**  
- **Hamza**  
- **Zhuofu**  

---

## **Project Overview**  
### **Objective**  
The goal of this project is to develop an anomaly detection model that leverages sentiment signals and high-frequency market data to identify and predict sudden market shifts (e.g., flash crashes).  

### **Key Features**  
1. **Sentiment Analysis Pipeline**:  
   - Utilize pre-trained NLP models (e.g., FinBERT) for sentiment classification of textual data.  
   - Analyze the impact of sentiment shifts on financial markets over time.  

2. **High-Frequency Data Integration**:  
   - Combine minute-level or tick-level market data with real-time sentiment signals.  
   - Address asynchronous timestamps and lagged relationships between data sources.  

3. **Predictive Analysis**:  
   - Correlate sentiment trends with market anomalies.  
   - Build a risk-scoring mechanism for assets based on sentiment-driven signals.  

---

## **Data Sources**  
### 1. **Textual Data**  
- **News**:  
  - Google News API for real-time news headlines.  
  - Yahoo Finance for additional financial insights. [Yahoo Finance API Guide](https://algotrading101.com/learn/yahoo-finance-api-guide/).  

- **Social Media**:  
  - Twitter API to fetch tweets with relevant hashtags and keywords (e.g., #stocks, #marketcrash).  

### 2. **Financial Data**  
- Tick-by-tick and minute-level data sourced from:  
  - Yahoo Finance API  
  - Alpha Vantage  
  - Bloomberg Terminal (if available)  

---

## **Mathematical Framework**  
1. **Correlation Analysis**:  
   - Compute correlations between sentiment shifts and market behavior.  
   - Explore lagged effects to address time asynchrony.  

2. **Response Functions**:  
   - Quantify how sentiment signals influence price dynamics.  

3. **Hayashi-Yoshida Cumulative Covariance Estimator**:  
   - Use this method to estimate covariances between high-frequency financial data and asynchronous sentiment data.  
   - This estimator is specifically designed for irregularly spaced data and can handle situations where timestamps differ between sentiment and price movements.

4. **True/False Discovery Rates**:  
   - Evaluate the accuracy of anomaly detection using predictive metrics.  

---

## **Project Structure**  
- `data/`: Raw and processed datasets.  
- `notebooks/`: Jupyter notebooks for development and experiments.  
- `scripts/`: Python scripts for data processing and model building.  
- `tests/`: Unit tests for the codebase.  
- `docs/`: Documentation and reports.  

---

## **How to Run**  

### **Clone the Repository**  
```bash  
git clone https://github.com/<username>/Sentiment-Market-Anomaly.git  
cd Sentiment-Market-Anomaly  
