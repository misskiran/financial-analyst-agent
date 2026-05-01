# 📈 AI Financial Analyst Agent

An agentic AI workflow designed to perform automated stock research, sentiment analysis, and paper trading with a built-in Human-in-the-Loop approval system.

## 🚀 Overview

This agent leverages **LangGraph** to coordinate a multi-step financial analysis process. It fetches live news and market data, uses **Google Gemini** to synthesize a sentiment-based recommendation, and allows you to approve or reject trades directly from a **Streamlit** dashboard.

---

## 🛠️ Technology Stack

- **Workflow Engine:** [LangGraph](https://github.com/langchain-ai/langgraph) (for stateful, cyclic graphs)
- **AI Brain:** [Google Gemini 1.5 Flash](https://aistudio.google.com/) (Sentiment analysis & decision making)
- **Search Tool:** [Tavily AI](https://tavily.com/) (Real-time financial news search)
- **Market Data:** [yfinance](https://github.com/ranaroussi/yfinance) (Live stock prices and P/E ratios)
- **Trading API:** [Alpaca Markets](https://alpaca.markets/) (Paper trading execution)
- **Frontend:** [Streamlit](https://streamlit.io/) (Interactive dashboard)

---

## 🔄 Agentic Workflow (The Process)

The agent moves through a defined set of "Nodes" in a specific order:

### 1. 📡 Research Node
- **News Search:** Calls Tavily to find the top 5 most relevant news headlines for the target ticker (e.g., "NVDA stock performance today").
- **Price Fetch:** Uses yfinance to retrieve the current market price and the trailing P/E ratio.
- **Timestamping:** Marks the exact time data was fetched to ensure fresh analysis.

### 2. 🧠 Analysis Node
- **Data Synthesis:** Consolidates news headlines and price metrics into a prompt.
- **LLM Reasoning:** Gemini analyzes the news context against the price data.
- **Recommendation:** Gemini outputs a JSON response containing a summary, sentiment (POSITIVE/NEGATIVE/NEUTRAL), and a suggested action (BUY/SELL/HOLD).

### 3. ⏸️ Human-in-the-Loop (Interrupt)
- **Stale Data Check:** If more than 10 minutes (configurable) have passed since the data was fetched, the agent forces a loop back to the **Research Node** to prevent trading on old news.
- **Approval Window:** The workflow pauses. The user must review the AI's summary and recommendation.
- **Decision:** The user clicks "Approve" to proceed to execution or "Reject" to end the run.

### 4. 🚀 Action Node
- **Execution:** If approved, the agent calls the Alpaca API to place a **Market Order** in your paper trading account.
- **Validation:** Captures the Order ID for tracking.

### 5. ✅ Complete
- Finalizes the state and displays a summary of the entire run including news, price, sentiment, and trade status.

---

## ⚙️ Setup & Execution

### 1. Requirements
Install dependencies using pip:
```bash
pip install langgraph langchain-google-genai tavily-python yfinance alpaca-trade-api python-dotenv streamlit
```

### 2. Environment Variables
Create a `.env` file with your API keys:
```env
TAVILY_API_KEY=your_key
GOOGLE_API_KEY=your_key
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 3. Running the App
Launch the interactive dashboard:
```bash
streamlit run app.py
```

---

## 🛡️ Safety Features
- **Human-in-the-Loop:** No trades are ever placed without explicit manual approval in the UI.
- **Stale Data Protection:** Automatically detects if market conditions might have changed since the start of the analysis and triggers a re-fetch.
- **Paper Trading:** Defaulted to Alpaca's paper API to ensure no real money is at risk during testing.
