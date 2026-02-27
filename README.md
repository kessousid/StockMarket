# Stock Market Prediction Tool (India + US)

A Streamlit web app that predicts **Buy / Hold / Sell** for Indian (NSE) and US (NYSE, NASDAQ) stocks using technical analysis, news sentiment analysis, and quarterly fundamental analysis.

## Features

- **Dual Market Support** — India (NSE) and US (NYSE / NASDAQ)
- **Technical Analysis** — SMA crossover, RSI, momentum signals
- **Sentiment Analysis** — Google News headlines scored with VADER
- **Fundamental Analysis** — QoQ revenue/profit growth, D/E ratio, current ratio, ROE
- **Key Metrics Dashboard** — P/E, P/B, ROE, ROCE, Piotroski F-Score, market cap
- **Stock Screener** — scan entire indices (S&P 500, Nifty 50, full NYSE/NSE, etc.)
- **Dynamic Stock Lists** — fetches live constituents for NSE (~2,600), NYSE (~2,700), NASDAQ (~4,000), S&P 500, NASDAQ 100, Dow 30

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git (to clone the repo)

## Setup & Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/kessousid/StockMarket.git
cd StockMarket
```

### 2. Create a virtual environment (recommended)

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run stock_predictor.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. **Select a market** — Choose India or US from the sidebar
2. **Pick a stock** — Browse by category/index/sector, or enter a custom ticker
3. **Click Analyze** — View the full prediction with technical, sentiment, and fundamental breakdowns
4. **Stock Screener** — Select a scope (e.g. S&P 500, All NSE Stocks) and click Run Screener to scan multiple stocks at once

## Project Structure

```
StockMarket/
├── stock_predictor.py   # Main application (single file)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web app framework |
| yfinance | Stock price and financial data |
| pandas | Data manipulation |
| plotly | Interactive charts |
| vaderSentiment | News headline sentiment scoring |
| feedparser | Google News RSS parsing |
| requests | HTTP requests |
| lxml | HTML table parsing (for US stock lists) |

## Disclaimer

This tool is for **educational purposes only**. Do not make investment decisions based solely on this analysis. Always consult a qualified financial advisor.
