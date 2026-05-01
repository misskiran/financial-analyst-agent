"""
Financial Analyst AI Agent
==========================
A LangGraph agentic workflow that:
  1. Searches for NVIDIA stock news via Tavily
  2. Fetches live price & P/E ratio via yfinance
  3. Asks Gemini to analyse sentiment
  4. Pauses for human approval (interrupt / human-in-the-loop)
  5. Places a paper trade via Alpaca if approved
  6. Detects stale data (> 10 min old) and loops back to re-fetch

Requires a .env file (or exported env vars) with:
  TAVILY_API_KEY
  GOOGLE_API_KEY          (Gemini)
  ALPACA_API_KEY
  ALPACA_SECRET_KEY
  ALPACA_BASE_URL         (e.g. https://paper-api.alpaca.markets)
"""

from dotenv import load_dotenv
load_dotenv()   

import os
import json
import time
from datetime import datetime
from typing import TypedDict, Annotated, Literal


from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


from tavily import TavilyClient
import yfinance as yf
import alpaca_trade_api as tradeapi

TICKER          = "NVDA"
TRADE_QTY       = 1                 # shares to buy / sell
STALE_MINUTES   = 10                # how old data can be before a re-fetch
# Strip trailing /v2 if present — the SDK appends it internally
_raw_url       = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
PAPER_BASE_URL = _raw_url.rstrip("/").removesuffix("/v2")

# ── Clients ───────────────────────────────────────────────────────────────────
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))

alpaca_client = tradeapi.REST(
    key_id     = os.getenv("ALPACA_API_KEY", ""),
    secret_key = os.getenv("ALPACA_SECRET_KEY", ""),
    base_url   = PAPER_BASE_URL,
)

llm = ChatGoogleGenerativeAI(
    model       = "gemini-1.5-flash",
    temperature = 0.2,
    google_api_key = os.getenv("GOOGLE_API_KEY", ""),
)

# ─────────────────────────────────────────────────────────────────────────────
# AGENT STATE
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    ticker:           str
    news_headlines:   list[str]
    current_price:    float | None
    pe_ratio:         float | None
    fetched_at:       float | None        # unix timestamp of last data fetch
    sentiment:        str | None          # "positive" | "negative" | "neutral"
    analysis_summary: str | None
    human_approved:   bool | None
    trade_action:     str | None          # "buy" | "sell" | None
    order_id:         str | None
    error:            str | None

# ─────────────────────────────────────────────────────────────────────────────
# TOOL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def search_news(ticker: str) -> list[str]:
    """Use Tavily to fetch recent news headlines for a given stock ticker."""
    query = f"{ticker} stock performance today news"
    print(f"\n[Tool: search_news] Searching Tavily for: '{query}' ...")
    try:
        results = tavily_client.search(query=query, max_results=5)
        headlines = [r["title"] for r in results.get("results", [])]
        print(f"[Tool: search_news] Found {len(headlines)} headlines.")
        for i, h in enumerate(headlines, 1):
            print(f"  {i}. {h}")
        return headlines
    except Exception as e:
        print(f"[Tool: search_news] ERROR: {e}")
        return []


def get_stock_price(ticker: str) -> dict:
    """Use yfinance to get the current price and trailing P/E ratio."""
    print(f"\n[Tool: get_stock_price] Fetching {ticker} data from yfinance ...")
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        pe    = info.get("trailingPE")
        print(f"[Tool: get_stock_price] Price: ${price}  |  P/E: {pe}")
        return {"price": price, "pe_ratio": pe}
    except Exception as e:
        print(f"[Tool: get_stock_price] ERROR: {e}")
        return {"price": None, "pe_ratio": None}


def place_order(ticker: str, side: Literal["buy", "sell"], qty: int = TRADE_QTY) -> str:
    """Use Alpaca paper API to place a market order."""
    print(f"\n[Tool: place_order] Submitting {side.upper()} order: {qty} x {ticker} ...")
    try:
        order = alpaca_client.submit_order(
            symbol     = ticker,
            qty        = qty,
            side       = side,
            type       = "market",
            time_in_force = "day",
        )
        order_id = order.id
        print(f"[Tool: place_order] Order submitted successfully. ID: {order_id}")
        return order_id
    except Exception as e:
        print(f"[Tool: place_order] ERROR: {e}")
        return f"ERROR: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# NODE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def research_node(state: AgentState) -> AgentState:
    """
    Research Node
    Fetches news headlines and live price data, then stamps the fetch time.
    """
    print("\n" + "="*60)
    print("📡  RESEARCH NODE — fetching live data")
    print("="*60)

    ticker    = state["ticker"]
    headlines = search_news(ticker)
    price_data = get_stock_price(ticker)

    return {
        **state,
        "news_headlines": headlines,
        "current_price":  price_data["price"],
        "pe_ratio":       price_data["pe_ratio"],
        "fetched_at":     time.time(),
        "error":          None,
    }


def analysis_node(state: AgentState) -> AgentState:
    """
    Analysis Node
    Asks Gemini to analyse news + price data and determine sentiment.
    """
    print("\n" + "="*60)
    print("🧠  ANALYSIS NODE — asking Gemini to analyse data")
    print("="*60)

    headlines = state["news_headlines"]
    price     = state["current_price"]
    pe        = state["pe_ratio"]
    ticker    = state["ticker"]

    headlines_str = "\n".join(f"- {h}" for h in headlines) if headlines else "No headlines found."

    prompt = f"""You are a financial analyst assistant. Analyse the following data for {ticker} and provide:
1. A brief analysis summary (2-3 sentences)
2. An overall sentiment classification: POSITIVE, NEGATIVE, or NEUTRAL
3. A recommended action: BUY, SELL, or HOLD

Current Data:
- Price: ${price}
- Trailing P/E Ratio: {pe}

Recent News Headlines:
{headlines_str}

Respond in this exact JSON format:
{{
  "summary": "...",
  "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "action": "BUY" | "SELL" | "HOLD"
}}"""

    print(f"[Analysis Node] Sending prompt to Gemini ...")
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        summary   = parsed.get("summary", "")
        sentiment = parsed.get("sentiment", "NEUTRAL").upper()
        action    = parsed.get("action", "HOLD").upper()

        print(f"\n[Analysis Node] --- Gemini Response ---")
        print(f"  Summary   : {summary}")
        print(f"  Sentiment : {sentiment}")
        print(f"  Action    : {action}")

        trade_action = None
        if action == "BUY":
            trade_action = "buy"
        elif action == "SELL":
            trade_action = "sell"

        return {
            **state,
            "sentiment":        sentiment,
            "analysis_summary": summary,
            "trade_action":     trade_action,
        }
    except Exception as e:
        print(f"[Analysis Node] ERROR parsing Gemini response: {e}")
        return {
            **state,
            "sentiment":        "NEUTRAL",
            "analysis_summary": "Analysis failed.",
            "trade_action":     None,
            "error":            str(e),
        }


def human_approval_node(state: AgentState) -> AgentState:
    """
    Human-in-the-Loop Interrupt Node
    Displays the analysis and waits for the user to type yes/no.
    """
    print("\n" + "="*60)
    print("⏸️   HUMAN APPROVAL REQUIRED")
    print("="*60)
    print(f"\n  Ticker    : {state['ticker']}")
    print(f"  Price     : ${state['current_price']}")
    print(f"  P/E Ratio : {state['pe_ratio']}")
    print(f"  Sentiment : {state['sentiment']}")
    print(f"  Summary   : {state['analysis_summary']}")

    action = state.get("trade_action")
    if action:
        print(f"\n  Proposed Action : {action.upper()} {TRADE_QTY} share(s) of {state['ticker']}")
    else:
        print(f"\n  Proposed Action : HOLD — no trade will be placed.")

    if not action:
        print("\n[Human Approval] Sentiment is HOLD/NEUTRAL — skipping trade. No approval needed.")
        return {**state, "human_approved": False}

    while True:
        user_input = input(f"\n  ▶ Approve {action.upper()} trade? [yes/no]: ").strip().lower()
        if user_input in ("yes", "y"):
            print("  ✅ Trade APPROVED by human.")
            return {**state, "human_approved": True}
        elif user_input in ("no", "n"):
            print("  ❌ Trade REJECTED by human.")
            return {**state, "human_approved": False}
        else:
            print("  Please type 'yes' or 'no'.")


def action_node(state: AgentState) -> AgentState:
    """
    Action Node
    Places the approved paper trade via Alpaca.
    """
    print("\n" + "="*60)
    print("🚀  ACTION NODE — executing trade")
    print("="*60)

    if not state.get("human_approved"):
        print("[Action Node] Trade not approved — skipping.")
        return state

    trade_action = state.get("trade_action")
    if not trade_action:
        print("[Action Node] No trade action defined — skipping.")
        return state

    order_id = place_order(state["ticker"], trade_action)
    return {**state, "order_id": order_id}


def end_node(state: AgentState) -> AgentState:
    """Final node — prints a run summary."""
    print("\n" + "="*60)
    print("✅  AGENT RUN COMPLETE")
    print("="*60)
    print(f"  Ticker        : {state['ticker']}")
    print(f"  Sentiment     : {state['sentiment']}")
    print(f"  Approved      : {state.get('human_approved')}")
    print(f"  Trade Action  : {state.get('trade_action', 'none')}")
    print(f"  Order ID      : {state.get('order_id', 'N/A')}")
    if state.get("error"):
        print(f"  Error         : {state['error']}")
    print("="*60 + "\n")
    return state

# ─────────────────────────────────────────────────────────────────────────────
# EDGE / ROUTING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def route_after_analysis(state: AgentState) -> str:
    """
    After analysis, check if the fetched data has gone stale (> STALE_MINUTES).
    If so, loop back to the research node. Otherwise continue to human approval.
    """
    fetched_at = state.get("fetched_at")
    if fetched_at is not None:
        elapsed_minutes = (time.time() - fetched_at) / 60
        if elapsed_minutes > STALE_MINUTES:
            print(f"\n⚠️  [Router] Data is {elapsed_minutes:.1f} min old (> {STALE_MINUTES} min limit). "
                  f"Looping back to research node to refresh ...")
            return "research"
    return "human_approval"


def route_after_approval(state: AgentState) -> str:
    """After human approval decision, route to action or end."""
    if state.get("human_approved"):
        return "action"
    return "end"

# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE LANGGRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("research",       research_node)
    graph.add_node("analysis",       analysis_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("action",         action_node)
    graph.add_node("end",            end_node)

    # Entry point
    graph.set_entry_point("research")

    # Edges
    graph.add_edge("research", "analysis")

    # Conditional: stale data check after analysis
    graph.add_conditional_edges(
        "analysis",
        route_after_analysis,
        {
            "research":       "research",
            "human_approval": "human_approval",
        },
    )

    # Conditional: approved → action, rejected → end
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "action": "action",
            "end":    "end",
        },
    )

    graph.add_edge("action", "end")
    graph.add_edge("end", END)

    return graph

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "🏦 " * 20)
    print("  FINANCIAL ANALYST AI AGENT")
    print("  Powered by LangGraph + Gemini + Tavily + Alpaca")
    print("🏦 " * 20)

    # Initial state
    initial_state: AgentState = {
        "ticker":           TICKER,
        "news_headlines":   [],
        "current_price":    None,
        "pe_ratio":         None,
        "fetched_at":       None,
        "sentiment":        None,
        "analysis_summary": None,
        "human_approved":   None,
        "trade_action":     None,
        "order_id":         None,
        "error":            None,
    }

    # Build and compile the graph
    graph = build_graph()
    compiled = graph.compile()

    # Run the agent
    final_state = compiled.invoke(initial_state)

    print("\nFinal state summary:")
    for k, v in final_state.items():
        if k != "news_headlines":
            print(f"  {k}: {v}")
    print(f"  news_headlines ({len(final_state.get('news_headlines', []))} items)")


if __name__ == "__main__":
    main()
