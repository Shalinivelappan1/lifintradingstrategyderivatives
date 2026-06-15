import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import random

st.set_page_config(
    page_title="Derivatives Trading Strategies Lab",
    page_icon="📐",
    layout="wide"
)

# ── BSM helpers ──────────────────────────────────────────
def bsm(S, K, T, r, sigma, opt="call"):
    if T <= 0 or sigma <= 0:
        return max(S-K,0) if opt=="call" else max(K-S,0)
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt=="call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def greeks(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return {"dc":1.0,"dp":-1.0,"gamma":0,"vega":0,"tc":0,"tp":0,"rc":0,"rp":0}
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    dc=norm.cdf(d1); dp=dc-1
    gamma=norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega=S*norm.pdf(d1)*np.sqrt(T)/100
    tc=(-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365
    tp=(-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    rc=K*T*np.exp(-r*T)*norm.cdf(d2)/100
    rp=-K*T*np.exp(-r*T)*norm.cdf(-d2)/100
    return {"dc":dc,"dp":dp,"gamma":gamma,"vega":vega,"tc":tc,"tp":tp,"rc":rc,"rp":rp}

def spot_range(S, pct=0.15, n=300):
    return np.linspace(S*(1-pct), S*(1+pct), n)

def pct_fmt(x,d=2): return f"{round(x,d)}%"
def cr(x): return f"₹{x:,.2f}"

# ── Header ───────────────────────────────────────────────
st.title("📐 Derivatives Trading Strategies Lab")
st.markdown("""
Master every major derivatives strategy — payoff diagrams, breakeven analysis,
BSM-priced scenarios, and real Indian market examples.

Covered: ✅ Directional ✅ Volatility ✅ Income ✅ Hedging ✅ Structured
""")

menu = st.sidebar.radio("Choose Strategy", [
    "── DIRECTIONAL ──",
    "Long Call",
    "Short Call (Naked)",
    "Long Put",
    "Short Put (Naked)",
    "Bull Call Spread",
    "Bear Put Spread",
    "Bull Put Spread (Credit)",
    "Bear Call Spread (Credit)",
    "Long Futures",
    "Short Futures",
    "── VOLATILITY ──",
    "Long Straddle",
    "Short Straddle",
    "Long Strangle",
    "Short Strangle",
    "Long Butterfly",
    "Short Butterfly",
    "Long Condor",
    "Calendar Spread",
    "Diagonal Spread",
    "── INCOME & HEDGING ──",
    "Covered Call",
    "Protective Put",
    "Collar",
    "Cash-Secured Put",
    "Married Put",
    "── ADVANCED ──",
    "Ratio Call Spread",
    "Ratio Put Spread",
    "Backspread (Call)",
    "Backspread (Put)",
    "Strip",
    "Strap",
    "Jade Lizard",
    "Iron Condor",
    "Iron Butterfly",
    "── TOOLS ──",
    "Strategy Selector",
    "Multi-Strategy Payoff Comparator",
    "Greeks Dashboard",
    "Implied Volatility Surface",
    "Strategy P&L Simulator",
    "Quiz Engine",
    "Formula Cheat Sheet",
])

# ── Common sidebar inputs ─────────────────────────────────
if menu not in ["── DIRECTIONAL ──","── VOLATILITY ──","── INCOME & HEDGING ──","── ADVANCED ──","── TOOLS ──"]:
    st.sidebar.divider()
    st.sidebar.markdown("**Market Parameters**")
    S = st.sidebar.number_input("Spot Price S (₹)", value=22000.0, step=50.0)
    r = st.sidebar.number_input("Risk-free Rate r (%)", value=7.0) / 100
    sigma = st.sidebar.number_input("Implied Volatility σ (%)", value=18.0) / 100
    T_days = st.sidebar.number_input("Days to Expiry", value=30, min_value=1, max_value=365)
    T = T_days / 365
    lots = st.sidebar.number_input("Lots (1 lot=25 units)", value=1, min_value=1)
    lot = 25 * lots
    x = spot_range(S)

# ── Helper: strategy card ─────────────────────────────────
def strategy_card(name, view, construction, max_profit, max_loss, bep_list, use_when, avoid_when):
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown(f"### {name}")
        st.markdown(f"**Market View:** {view}")
        st.markdown(f"**Construction:** {construction}")
        st.info(f"🎯 **Use when:** {use_when}")
        st.warning(f"⚠️ **Avoid when:** {avoid_when}")
    with col2:
        kpis = {"Max Profit": max_profit, "Max Loss": max_loss}
        for bep_i, bep in enumerate(bep_list):
            kpis[f"BEP {'Upper' if bep_i==1 else ('Lower' if len(bep_list)>1 and bep_i==0 else '')}"] = bep
        for k, v in kpis.items():
            st.metric(k, v)

def payoff_chart(x, strategies, title, S=None, show_cost=True):
    fig = go.Figure()
    colors = ['#174EA6','#C03B3B','#157A42','#F5A623','#6A0DAD','#02B4AC']
    for i, (name, pnl, color) in enumerate(strategies):
        fig.add_trace(go.Scatter(x=x, y=pnl, mode='lines',
                                  name=name, line=dict(color=color or colors[i%len(colors)], width=2.5)))
    fig.add_hline(y=0, line_color='black', line_width=1)
    if S:
        fig.add_vline(x=S, line_dash='dash', line_color='gray',
                      annotation_text=f'Spot ₹{S:.0f}', annotation_position='top right')
    fig.update_layout(title=title, xaxis_title="Spot at Expiry (₹)",
                      yaxis_title="P&L per lot (₹)", height=400,
                      legend=dict(x=0, y=1))
    return fig

def pnl_table(x_pts, strategies, labels=None):
    df = {"Spot ST": [round(xi,0) for xi in x_pts]}
    for name, pnl_fn in strategies:
        df[name] = [round(pnl_fn(xi)*lot, 2) for xi in x_pts]
    return pd.DataFrame(df)

# ═══════════════════════════════════════════════════════════
# SECTION HEADERS
if menu in ["── DIRECTIONAL ──","── VOLATILITY ──","── INCOME & HEDGING ──","── ADVANCED ──","── TOOLS ──"]:
    icons = {"── DIRECTIONAL ──":"📈","── VOLATILITY ──":"🌪️","── INCOME & HEDGING ──":"🛡️","── ADVANCED ──":"🔬","── TOOLS ──":"🛠️"}
    labels = {"── DIRECTIONAL ──":"Directional Strategies","── VOLATILITY ──":"Volatility Strategies",
              "── INCOME & HEDGING ──":"Income & Hedging Strategies","── ADVANCED ──":"Advanced Strategies","── TOOLS ──":"Strategy Tools"}
    st.header(f"{icons[menu]} {labels[menu]}")
    st.info("Select a strategy from the sidebar to explore payoff diagrams, Greeks, breakevens, and Indian market examples.")

# ═══════════════════════════════════════════════════════════
elif menu == "Long Call":
    st.header("📈 Long Call")
    C = st.number_input("Call Premium C (₹)", value=round(bsm(S,S,T,r,sigma,"call"),2))
    K = st.number_input("Strike K (₹)", value=float(S))
    strategy_card("Long Call","Bullish — expect significant upward move",
        f"Buy 1 Call (K={K}) @ ₹{C}",
        "Unlimited (as Nifty rises)", f"₹{C*lot:,.0f} (premium paid)",
        [f"₹{K+C:.2f}"], "Before earnings / budget when expecting big up-move",
        "In high IV — premium too expensive; prefer spread instead")
    pnl_fn = lambda st_: max(st_-K,0) - C
    x_plot = spot_range(K)
    pnl = np.array([pnl_fn(xi)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Long Call P&L", pnl, '#174EA6')], "Long Call Payoff", S=K)
    fig.add_vline(x=K+C, line_dash='dot', line_color='green',
                  annotation_text=f'BEP ₹{K+C:.0f}')
    st.plotly_chart(fig, use_container_width=True)
    g = greeks(S,K,T,r,sigma)
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("BSM Fair Value",cr(bsm(S,K,T,r,sigma,"call")))
    col2.metric("Delta",round(g["dc"],4))
    col3.metric("Gamma",round(g["gamma"],6))
    col4.metric("Theta/day",cr(g["tc"]))
    st.info(f"**Indian Example:** Nifty at 22000. Buy 22000CE at ₹{C:.0f}. BEP = ₹{K+C:.0f}. "
            f"For every 100-point rise above BEP, profit = ₹{100*lot:,}.")

elif menu == "Short Call (Naked)":
    st.header("📉 Short Call (Naked Write)")
    K = st.number_input("Strike K (₹)", value=float(S*1.02))
    C = round(bsm(S,K,T,r,sigma,"call"),2)
    C_inp = st.number_input("Call Premium Received (₹)", value=C)
    strategy_card("Short Call (Naked)","Bearish to Neutral — expect market to stay below strike",
        f"Sell 1 Call (K={K}) @ ₹{C_inp}",
        f"₹{C_inp*lot:,.0f} (premium collected)", "UNLIMITED — Nifty can rise indefinitely",
        [f"₹{K+C_inp:.2f}"],"When IV is very high (VIX>25) and you expect it to fall",
        "Before major event / budget — risk of sudden spike")
    x_plot = spot_range(K)
    pnl = np.array([(C_inp - max(xi-K,0))*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Short Call P&L", pnl, '#C03B3B')], "Short Call Payoff", S=K)
    fig.add_vline(x=K+C_inp, line_dash='dot', line_color='red',
                  annotation_text=f'BEP ₹{K+C_inp:.0f}')
    st.plotly_chart(fig, use_container_width=True)
    st.error("⚠️ UNLIMITED LOSS RISK. Never hold naked short calls without a stop-loss or hedge.")

elif menu == "Long Put":
    st.header("📉 Long Put")
    K = st.number_input("Strike K (₹)", value=float(S))
    P = st.number_input("Put Premium P (₹)", value=round(bsm(S,K,T,r,sigma,"put"),2))
    strategy_card("Long Put","Bearish — expect significant fall",
        f"Buy 1 Put (K={K}) @ ₹{P}",
        f"₹{(K-P)*lot:,.0f} (if Nifty→0)", f"₹{P*lot:,.0f} (premium paid)",
        [f"₹{K-P:.2f}"],"Before bad news event or in high-beta portfolio hedging",
        "In high IV — use put spread instead to reduce premium cost")
    x_plot = spot_range(K)
    pnl = np.array([(max(K-xi,0)-P)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Long Put P&L", pnl, '#C03B3B')], "Long Put Payoff", S=K)
    fig.add_vline(x=K-P, line_dash='dot', line_color='orange',
                  annotation_text=f'BEP ₹{K-P:.0f}')
    st.plotly_chart(fig, use_container_width=True)
    g = greeks(S,K,T,r,sigma)
    col1,col2,col3,col4=st.columns(4)
    col1.metric("BSM Fair Value",cr(bsm(S,K,T,r,sigma,"put")))
    col2.metric("Delta",round(g["dp"],4))
    col3.metric("Gamma",round(g["gamma"],6))
    col4.metric("Theta/day",cr(g["tp"]))

elif menu == "Short Put (Naked)":
    st.header("📈 Short Put (Naked Write)")
    K = st.number_input("Strike K (₹)", value=float(S*0.97))
    P = round(bsm(S,K,T,r,sigma,"put"),2)
    P_inp = st.number_input("Put Premium Received (₹)", value=P)
    strategy_card("Short Put (Naked)","Neutral to Bullish — expect market to stay above strike",
        f"Sell 1 Put (K={K}) @ ₹{P_inp}",
        f"₹{P_inp*lot:,.0f} (premium)", f"₹{(K-P_inp)*lot:,.0f} (if Nifty→0)",
        [f"₹{K-P_inp:.2f}"],"When you'd be willing to BUY the underlying at strike (cash-secured)",
        "In weak or uncertain markets — downside can be large")
    x_plot = spot_range(K)
    pnl = np.array([(P_inp - max(K-xi,0))*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Short Put P&L", pnl, '#157A42')], "Short Put Payoff", S=K)
    fig.add_vline(x=K-P_inp, line_dash='dot', line_color='red', annotation_text=f'BEP ₹{K-P_inp:.0f}')
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Bull Call Spread":
    st.header("📈 Bull Call Spread (Debit Spread)")
    col1,col2 = st.columns(2)
    with col1:
        K1 = st.number_input("Lower Strike K₁ (₹)", value=float(S))
        C1 = st.number_input("Buy Call Premium @ K₁ (₹)", value=round(bsm(S,S,T,r,sigma,"call"),2))
    with col2:
        K2 = st.number_input("Upper Strike K₂ (₹)", value=float(S*1.02))
        C2 = st.number_input("Sell Call Premium @ K₂ (₹)", value=round(bsm(S,S*1.02,T,r,sigma,"call"),2))
    net = C1-C2
    max_p = (K2-K1)-net; bep1 = K1+net
    strategy_card("Bull Call Spread","Moderately Bullish — limited upside expected",
        f"Buy K₁={K1} Call @ ₹{C1:.2f} + Sell K₂={K2} Call @ ₹{C2:.2f}",
        cr(max_p*lot), cr(net*lot), [cr(bep1)],
        "When moderately bullish; reduces premium cost vs outright call",
        "When expecting a very large rally — upside is capped at K₂")
    x_plot = spot_range(K1, pct=0.06)
    pnl = np.array([(max(xi-K1,0)-max(xi-K2,0)-net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Bull Call Spread",pnl,'#174EA6')], "Bull Call Spread Payoff", S=S)
    fig.add_vline(x=bep1,line_dash='dot',line_color='green',annotation_text=f'BEP ₹{bep1:.0f}')
    fig.add_vline(x=K1,line_dash='dot',line_color='gray',annotation_text=f'K₁={K1:.0f}')
    fig.add_vline(x=K2,line_dash='dot',line_color='gray',annotation_text=f'K₂={K2:.0f}')
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Net Debit = ₹{net:.2f} | Max Profit = ₹{max_p:.2f} | BEP = ₹{bep1:.2f}\n"
            f"**Indian Example:** Nifty 22000. Buy 22000CE@{C1:.0f}, Sell 22400CE@{C2:.0f}. "
            f"Net cost ₹{net*lot:,.0f}. Max profit ₹{max_p*lot:,.0f} if Nifty ≥ {K2:.0f}.")

elif menu == "Bear Put Spread":
    st.header("📉 Bear Put Spread (Debit Spread)")
    col1,col2 = st.columns(2)
    with col1:
        K1 = st.number_input("Higher Strike K₁ (₹)", value=float(S))
        P1 = st.number_input("Buy Put Premium @ K₁ (₹)", value=round(bsm(S,S,T,r,sigma,"put"),2))
    with col2:
        K2 = st.number_input("Lower Strike K₂ (₹)", value=float(S*0.98))
        P2 = st.number_input("Sell Put Premium @ K₂ (₹)", value=round(bsm(S,S*0.98,T,r,sigma,"put"),2))
    net = P1-P2; max_p=(K1-K2)-net; bep1=K1-net
    strategy_card("Bear Put Spread","Moderately Bearish",
        f"Buy K₁={K1} Put @ ₹{P1:.2f} + Sell K₂={K2} Put @ ₹{P2:.2f}",
        cr(max_p*lot), cr(net*lot), [cr(bep1)],
        "Moderately bearish; cheaper than buying put outright",
        "Expecting a massive crash — downside capped at K₂")
    x_plot = spot_range(K1,pct=0.06)
    pnl = np.array([(max(K1-xi,0)-max(K2-xi,0)-net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Bear Put Spread",pnl,'#C03B3B')],"Bear Put Spread Payoff",S=S)
    fig.add_vline(x=bep1,line_dash='dot',line_color='orange',annotation_text=f'BEP ₹{bep1:.0f}')
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Bull Put Spread (Credit)":
    st.header("💰 Bull Put Spread (Credit Spread)")
    col1,col2 = st.columns(2)
    with col1:
        K1 = st.number_input("Sell Put Strike K₁ (₹)", value=float(S*0.98))
        P1 = st.number_input("Put Premium Received @ K₁ (₹)", value=round(bsm(S,S*0.98,T,r,sigma,"put"),2))
    with col2:
        K2 = st.number_input("Buy Put Strike K₂ (₹)", value=float(S*0.96))
        P2 = st.number_input("Put Premium Paid @ K₂ (₹)", value=round(bsm(S,S*0.96,T,r,sigma,"put"),2))
    net_credit = P1-P2; max_loss=(K1-K2)-net_credit; bep1=K1-net_credit
    strategy_card("Bull Put Spread","Neutral to Bullish — COLLECT premium",
        f"Sell K₁={K1} Put @ ₹{P1:.2f} + Buy K₂={K2} Put @ ₹{P2:.2f}",
        cr(net_credit*lot), cr(max_loss*lot), [cr(bep1)],
        "When mildly bullish and want premium income with defined risk",
        "When sharply bearish — loss can be meaningful")
    x_plot = spot_range(K2,pct=0.06)
    pnl = np.array([(net_credit-max(K1-xi,0)+max(K2-xi,0))*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Bull Put Spread",pnl,'#157A42')],"Bull Put Credit Spread",S=S)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Collect ₹{net_credit*lot:,.0f} upfront. Keep all of it if Nifty stays above ₹{K1:.0f}.")

elif menu == "Bear Call Spread (Credit)":
    st.header("💰 Bear Call Spread (Credit Spread)")
    col1,col2 = st.columns(2)
    with col1:
        K1 = st.number_input("Sell Call Strike K₁ (₹)", value=float(S*1.02))
        C1 = st.number_input("Call Premium Received @ K₁ (₹)", value=round(bsm(S,S*1.02,T,r,sigma,"call"),2))
    with col2:
        K2 = st.number_input("Buy Call Strike K₂ (₹)", value=float(S*1.04))
        C2 = st.number_input("Call Premium Paid @ K₂ (₹)", value=round(bsm(S,S*1.04,T,r,sigma,"call"),2))
    net_credit = C1-C2; max_loss=(K2-K1)-net_credit; bep1=K1+net_credit
    strategy_card("Bear Call Spread","Neutral to Bearish — COLLECT premium",
        f"Sell K₁={K1} Call + Buy K₂={K2} Call",
        cr(net_credit*lot), cr(max_loss*lot), [cr(bep1)],
        "When mildly bearish; collect premium with limited risk",
        "When sharply bullish expected — capped loss but painful")
    x_plot = spot_range(K1,pct=0.06)
    pnl = np.array([(net_credit-max(xi-K1,0)+max(xi-K2,0))*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Bear Call Spread",pnl,'#F5A623')],"Bear Call Credit Spread",S=S)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Long Futures":
    st.header("📈 Long Futures")
    F = st.number_input("Futures Entry Price (₹)", value=float(S+50))
    strategy_card("Long Futures","Bullish — full directional exposure",
        f"Buy 1 Nifty Futures @ ₹{F}","Unlimited","Unlimited (Nifty can fall to 0)",[f"₹{F:.2f}"],
        "Strong directional conviction; cost-effective vs buying stock",
        "When uncertain about direction — options give asymmetric protection")
    x_plot = spot_range(F)
    pnl = np.array([(xi-F)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Long Futures",pnl,'#174EA6')],"Long Futures P&L",S=F)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"P&L = (Nifty − {F:.0f}) × 25 per lot. Every 100-point move = ₹{100*lot:,} gain/loss.")

elif menu == "Short Futures":
    st.header("📉 Short Futures")
    F = st.number_input("Futures Entry Price (₹)", value=float(S-50))
    strategy_card("Short Futures","Bearish — full directional short exposure",
        f"Sell 1 Nifty Futures @ ₹{F}","Unlimited (Nifty can fall to 0)","Unlimited",[f"₹{F:.2f}"],
        "Strong bearish conviction; hedging long equity portfolio",
        "In bullish market — risk of unlimited loss if market rallies")
    x_plot = spot_range(F)
    pnl = np.array([(F-xi)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Short Futures",pnl,'#C03B3B')],"Short Futures P&L",S=F)
    st.plotly_chart(fig, use_container_width=True)

# ── VOLATILITY ────────────────────────────────────────────
elif menu == "Long Straddle":
    st.header("🌪️ Long Straddle")
    K = st.number_input("Strike K (ATM recommended)", value=float(S))
    C = round(bsm(S,K,T,r,sigma,"call"),2)
    P = round(bsm(S,K,T,r,sigma,"put"),2)
    C_inp = st.number_input("Call Premium C (₹)", value=C)
    P_inp = st.number_input("Put Premium P (₹)", value=P)
    net = C_inp+P_inp; bep_up=K+net; bep_dn=K-net
    strategy_card("Long Straddle","Volatile — expect BIG move, direction unknown",
        f"Buy ATM Call @ ₹{C_inp:.2f} + Buy ATM Put @ ₹{P_inp:.2f}",
        "Unlimited","Net premium = ₹"+cr(net*lot),[cr(bep_dn),cr(bep_up)],
        "Before earnings, budget, RBI policy — big binary event expected",
        "In low-volatility quiet markets — premium wasted to theta decay")
    x_plot = spot_range(K,pct=0.10)
    pnl = np.array([(max(xi-K,0)+max(K-xi,0)-net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Long Straddle",pnl,'#6A0DAD')],"Long Straddle Payoff",S=K)
    fig.add_vline(x=bep_up,line_dash='dot',line_color='green',annotation_text=f'BEP↑ ₹{bep_up:.0f}')
    fig.add_vline(x=bep_dn,line_dash='dot',line_color='green',annotation_text=f'BEP↓ ₹{bep_dn:.0f}')
    fig.add_hline(y=-net*lot,line_dash='dot',line_color='red',annotation_text=f'Max Loss ₹{net*lot:.0f}')
    st.plotly_chart(fig, use_container_width=True)
    col1,col2,col3 = st.columns(3)
    col1.metric("Net Cost",cr(net*lot)); col2.metric("Upper BEP",cr(bep_up)); col3.metric("Lower BEP",cr(bep_dn))
    col1.metric("Break-even Range",f"₹{round(bep_up-bep_dn,0):.0f}")
    col2.metric("Break-even % move",pct_fmt(net/K*100))
    st.info(f"**Indian Example:** Nifty {K:.0f} before Infosys results. Straddle costs ₹{net:.2f}. "
            f"Need Nifty to move more than {pct_fmt(net/K*100)} in either direction to profit.")

elif menu == "Short Straddle":
    st.header("💤 Short Straddle")
    K = st.number_input("Strike K", value=float(S))
    C_inp = st.number_input("Call Premium Received (₹)", value=round(bsm(S,K,T,r,sigma,"call"),2))
    P_inp = st.number_input("Put Premium Received (₹)", value=round(bsm(S,K,T,r,sigma,"put"),2))
    net = C_inp+P_inp; bep_up=K+net; bep_dn=K-net
    strategy_card("Short Straddle","Range-bound — expect low volatility, flat market",
        f"Sell ATM Call + Sell ATM Put @ K={K}",
        cr(net*lot),"UNLIMITED both directions",[cr(bep_dn),cr(bep_up)],
        "After a big event (IV crush), or when VIX is very high and expected to fall",
        "Before events — tail risk is extreme; one gap move destroys months of premium")
    x_plot = spot_range(K,pct=0.10)
    pnl = np.array([(net-max(xi-K,0)-max(K-xi,0))*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Short Straddle",pnl,'#C03B3B')],"Short Straddle Payoff",S=K)
    fig.add_vline(x=bep_up,line_dash='dot',line_color='red',annotation_text=f'BEP↑ ₹{bep_up:.0f}')
    fig.add_vline(x=bep_dn,line_dash='dot',line_color='red',annotation_text=f'BEP↓ ₹{bep_dn:.0f}')
    st.plotly_chart(fig, use_container_width=True)
    st.error("⚠️ UNLIMITED LOSS if market makes a large move. Always use stop-loss levels.")

elif menu == "Long Strangle":
    st.header("🌪️ Long Strangle")
    col1,col2 = st.columns(2)
    with col1:
        K1 = st.number_input("Put Strike K₁ (OTM)", value=float(S*0.97))
        P_inp = st.number_input("Put Premium (₹)", value=round(bsm(S,S*0.97,T,r,sigma,"put"),2))
    with col2:
        K2 = st.number_input("Call Strike K₂ (OTM)", value=float(S*1.03))
        C_inp = st.number_input("Call Premium (₹)", value=round(bsm(S,S*1.03,T,r,sigma,"call"),2))
    net = C_inp+P_inp; bep_up=K2+net; bep_dn=K1-net
    strategy_card("Long Strangle","Very Volatile — expect EXTREME move",
        f"Buy OTM Put (K₁={K1}) + Buy OTM Call (K₂={K2})",
        "Unlimited",cr(net*lot),[cr(bep_dn),cr(bep_up)],
        "When expecting extreme move; cheaper than straddle but needs bigger move",
        "In quiet markets — both legs decay rapidly to zero")
    x_plot = spot_range(S,pct=0.12)
    pnl = np.array([(max(xi-K2,0)+max(K1-xi,0)-net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Long Strangle",pnl,'#6A0DAD')],"Long Strangle Payoff",S=S)
    fig.add_vline(x=bep_up,line_dash='dot',line_color='green',annotation_text=f'BEP↑ ₹{bep_up:.0f}')
    fig.add_vline(x=bep_dn,line_dash='dot',line_color='green',annotation_text=f'BEP↓ ₹{bep_dn:.0f}')
    st.plotly_chart(fig, use_container_width=True)
    col1,col2,col3 = st.columns(3)
    col1.metric("Net Cost",cr(net*lot)); col2.metric("Upper BEP",cr(bep_up)); col3.metric("Lower BEP",cr(bep_dn))

elif menu == "Short Strangle":
    st.header("💤 Short Strangle")
    col1,col2 = st.columns(2)
    with col1:
        K1 = st.number_input("Sell Put Strike K₁", value=float(S*0.97))
        P_inp = st.number_input("Put Premium Received (₹)", value=round(bsm(S,S*0.97,T,r,sigma,"put"),2))
    with col2:
        K2 = st.number_input("Sell Call Strike K₂", value=float(S*1.03))
        C_inp = st.number_input("Call Premium Received (₹)", value=round(bsm(S,S*1.03,T,r,sigma,"call"),2))
    net = C_inp+P_inp; bep_up=K2+net; bep_dn=K1-net
    strategy_card("Short Strangle","Range-bound — wider profit zone than straddle",
        f"Sell OTM Put (K₁={K1}) + Sell OTM Call (K₂={K2})",
        cr(net*lot),"UNLIMITED both directions",[cr(bep_dn),cr(bep_up)],
        "High IV environment; market expected to stay between K₁ and K₂",
        "Before events — tail risk is extreme")
    x_plot = spot_range(S,pct=0.12)
    pnl = np.array([(net-max(xi-K2,0)-max(K1-xi,0))*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Short Strangle",pnl,'#F5A623')],"Short Strangle Payoff",S=S)
    fig.add_vline(x=bep_up,line_dash='dot',line_color='red',annotation_text=f'BEP↑ ₹{bep_up:.0f}')
    fig.add_vline(x=bep_dn,line_dash='dot',line_color='red',annotation_text=f'BEP↓ ₹{bep_dn:.0f}')
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Profit zone: ₹{K1:.0f} to ₹{K2:.0f}. Collect ₹{net*lot:,.0f} upfront.")

elif menu == "Long Butterfly":
    st.header("🦋 Long Butterfly Spread")
    col1,col2,col3 = st.columns(3)
    with col1:
        K1 = st.number_input("K₁ (Lower)", value=float(S-300))
        C1 = st.number_input("Buy Call @ K₁ (₹)", value=round(bsm(S,S-300,T,r,sigma,"call"),2))
    with col2:
        K2 = st.number_input("K₂ (Middle ATM)", value=float(S))
        C2 = st.number_input("Sell 2× Call @ K₂ (₹)", value=round(bsm(S,S,T,r,sigma,"call"),2))
    with col3:
        K3 = st.number_input("K₃ (Upper)", value=float(S+300))
        C3 = st.number_input("Buy Call @ K₃ (₹)", value=round(bsm(S,S+300,T,r,sigma,"call"),2))
    net = C1-2*C2+C3; max_p=(K2-K1)-net; bep_up=K3-net; bep_dn=K1+net
    strategy_card("Long Butterfly","Neutral — expect price to stay near K₂",
        f"Buy K₁ Call + Sell 2×K₂ Call + Buy K₃ Call",
        cr(max_p*lot),cr(net*lot),[cr(bep_dn),cr(bep_up)],
        "Range-bound market; low cost; exam favourite strategy",
        "When expecting a breakout — all profit is in the middle")
    x_plot = spot_range(K2,pct=0.08)
    pnl = np.array([(max(xi-K1,0)-2*max(xi-K2,0)+max(xi-K3,0)-net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Long Butterfly",pnl,'#02B4AC')],"Long Butterfly Payoff",S=K2)
    for k,lb in [(K1,'K₁'),(K2,'K₂ (peak)'),(K3,'K₃')]:
        fig.add_vline(x=k,line_dash='dot',line_color='gray',annotation_text=lb)
    st.plotly_chart(fig, use_container_width=True)
    col1,col2,col3 = st.columns(3)
    col1.metric("Net Cost",cr(net*lot)); col2.metric("Max Profit",cr(max_p*lot))
    col3.metric("Profit if Nifty at K₂",cr(max_p*lot))

elif menu == "Short Butterfly":
    st.header("🦋 Short Butterfly Spread")
    col1,col2,col3 = st.columns(3)
    with col1: K1=st.number_input("K₁",value=float(S-300)); C1=round(bsm(S,S-300,T,r,sigma,"call"),2)
    with col2: K2=st.number_input("K₂",value=float(S)); C2=round(bsm(S,S,T,r,sigma,"call"),2)
    with col3: K3=st.number_input("K₃",value=float(S+300)); C3=round(bsm(S,S+300,T,r,sigma,"call"),2)
    net = -(C1-2*C2+C3); max_p=net; max_l=(K2-K1)-net
    strategy_card("Short Butterfly","Volatile — expect price to break out from K₂",
        "Sell K₁ Call + Buy 2×K₂ Call + Sell K₃ Call",
        cr(max_p*lot),cr(max_l*lot),[cr(K1+max_p),cr(K3-max_p)],
        "When you expect a big breakout but are unsure which direction",
        "Range-bound market — max loss if price stays at K₂")
    x_plot = spot_range(K2,pct=0.08)
    pnl = np.array([(-max(xi-K1,0)+2*max(xi-K2,0)-max(xi-K3,0)+net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Short Butterfly",pnl,'#F5A623')],"Short Butterfly Payoff",S=K2)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Long Condor":
    st.header("🦅 Long Condor")
    col1,col2 = st.columns(2)
    with col1:
        K1=st.number_input("K₁",value=float(S-400)); C1=round(bsm(S,S-400,T,r,sigma,"call"),2)
        K2=st.number_input("K₂",value=float(S-100)); C2=round(bsm(S,S-100,T,r,sigma,"call"),2)
    with col2:
        K3=st.number_input("K₃",value=float(S+100)); C3=round(bsm(S,S+100,T,r,sigma,"call"),2)
        K4=st.number_input("K₄",value=float(S+400)); C4=round(bsm(S,S+400,T,r,sigma,"call"),2)
    net=C1-C2-C3+C4; max_p=(K2-K1)-net
    strategy_card("Long Condor","Neutral — WIDER profit zone than butterfly",
        "Buy K₁ Call + Sell K₂ Call + Sell K₃ Call + Buy K₄ Call",
        cr(max_p*lot),cr(net*lot),[cr(K1+net),cr(K4-net)],
        "Range-bound; wider band; lower risk than butterfly but also lower max profit",
        "When expecting a strong directional move")
    x_plot = spot_range(S,pct=0.10)
    pnl = np.array([(max(xi-K1,0)-max(xi-K2,0)-max(xi-K3,0)+max(xi-K4,0)-net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Long Condor",pnl,'#02B4AC')],"Long Condor Payoff",S=S)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Calendar Spread":
    st.header("📅 Calendar Spread (Time Spread)")
    st.markdown("""
## Calendar Spread — Exploit Time Decay Difference

**Construction:** Sell near-term option + Buy far-term option at SAME strike.

**Profit from:** Near-term option decays faster (higher theta) than far-term.

**View:** Neutral in near-term, possibly directional longer-term.

**Key insight:** Near-term ATM theta >> Far-term ATM theta.
""")
    K = st.number_input("Strike K (₹)", value=float(S))
    T1 = st.number_input("Near-term expiry (days)", value=7)
    T2 = st.number_input("Far-term expiry (days)", value=30)
    near_call = bsm(S,K,T1/365,r,sigma,"call")
    far_call = bsm(S,K,T2/365,r,sigma,"call")
    net_cost = far_call - near_call
    col1,col2,col3 = st.columns(3)
    col1.metric("Near Call (Sell)",cr(near_call))
    col2.metric("Far Call (Buy)",cr(far_call))
    col3.metric("Net Debit",cr(net_cost))
    st.info(f"""
**How it works:**
- Sell {T1}-day {K:.0f} Call @ ₹{near_call:.2f}
- Buy {T2}-day {K:.0f} Call @ ₹{far_call:.2f}
- Net cost = ₹{net_cost:.2f} per unit = ₹{net_cost*lot:,.0f} per lot

**Profit if:** Nifty stays near ₹{K:.0f} at near-term expiry.
Near-term call expires worthless → you keep premium.
Far-term call still has value.
""")
    # Near vs far theta
    near_theta = greeks(S,K,T1/365,r,sigma)["tc"]
    far_theta = greeks(S,K,T2/365,r,sigma)["tc"]
    st.info(f"Near-term daily theta: ₹{near_theta*lot:.2f}/lot | Far-term: ₹{far_theta*lot:.2f}/lot\nTheta advantage: ₹{(near_theta-far_theta)*lot:.2f}/lot/day")

elif menu == "Diagonal Spread":
    st.header("↗️ Diagonal Spread")
    st.markdown("""
## Diagonal Spread

**Construction:** Sell near-term option at one strike + Buy far-term option at DIFFERENT strike.

**A hybrid of:** Calendar spread (time) + Vertical spread (strike).

**Common form:** Buy far ITM call (low theta, high delta) + Sell near OTM call (high theta).

Net effect: Cheap way to get directional exposure while selling time decay.
""")
    col1,col2 = st.columns(2)
    with col1:
        K_near = st.number_input("Near-term Strike (Sell)", value=float(S*1.02))
        T_near = st.number_input("Near-term Expiry (days)", value=7)
        near_p = bsm(S,K_near,T_near/365,r,sigma,"call")
        st.metric("Near Call Premium (Receive)", cr(near_p))
    with col2:
        K_far = st.number_input("Far-term Strike (Buy)", value=float(S*0.99))
        T_far = st.number_input("Far-term Expiry (days)", value=30)
        far_p = bsm(S,K_far,T_far/365,r,sigma,"call")
        st.metric("Far Call Premium (Pay)", cr(far_p))
    net_diag = far_p - near_p
    st.metric("Net Cost", cr(net_diag))
    st.success(f"Net {'debit' if net_diag>0 else 'credit'} = ₹{abs(net_diag):.2f}/unit = ₹{abs(net_diag)*lot:,.0f}/lot")

# ── INCOME & HEDGING ──────────────────────────────────────
elif menu == "Covered Call":
    st.header("💰 Covered Call")
    stock_price = st.number_input("Stock/Futures Purchase Price (₹)", value=float(S))
    K = st.number_input("Call Strike to Sell K (₹)", value=float(S*1.02))
    C_recv = st.number_input("Call Premium Received (₹)", value=round(bsm(S,K,T,r,sigma,"call"),2))
    max_p = (K-stock_price)+C_recv; bep1=stock_price-C_recv
    strategy_card("Covered Call","Neutral to mildly bullish — generate income on existing holding",
        f"Long Stock @ ₹{stock_price:.0f} + Sell {K:.0f} Call @ ₹{C_recv:.2f}",
        cr(max_p*lot),f"₹{stock_price:.0f} − ₹{C_recv:.2f} = {cr(bep1)} (stock falls to 0)",[cr(bep1)],
        "Holding stock long-term; market is flat/mildly bullish",
        "When expecting big rally — upside is capped at K")
    x_plot = spot_range(stock_price,pct=0.08)
    pnl = np.array([(xi-stock_price + C_recv - max(xi-K,0))*lot for xi in x_plot])
    stock_pnl = np.array([(xi-stock_price)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[
        ("Covered Call",pnl,'#174EA6'),
        ("Stock Only",stock_pnl,'#B5CCFF')
    ],"Covered Call vs Stock Only Payoff",S=stock_price)
    fig.add_vline(x=K,line_dash='dot',line_color='gray',annotation_text=f'Cap at K=₹{K:.0f}')
    fig.add_vline(x=bep1,line_dash='dot',line_color='green',annotation_text=f'BEP ₹{bep1:.0f}')
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"If Nifty stays below ₹{K:.0f}, you keep full ₹{C_recv*lot:,.0f} premium — extra income on your position.")

elif menu == "Protective Put":
    st.header("🛡️ Protective Put (Portfolio Insurance)")
    stock_price = st.number_input("Stock Purchase Price (₹)", value=float(S))
    K = st.number_input("Put Strike K (₹)", value=float(S*0.97))
    P_paid = st.number_input("Put Premium Paid (₹)", value=round(bsm(S,K,T,r,sigma,"put"),2))
    max_loss = stock_price-K+P_paid; bep1=stock_price+P_paid
    strategy_card("Protective Put","Long stock but want crash protection",
        f"Long Stock @ ₹{stock_price:.0f} + Buy {K:.0f} Put @ ₹{P_paid:.2f}",
        "Unlimited (full upside)","CAPPED at ₹"+cr(max_loss*lot),[cr(bep1)],
        "Before earnings/budget when long stock and fearing gap-down",
        "In stable markets — put premium is wasted if no crash comes")
    x_plot = spot_range(stock_price,pct=0.10)
    pnl = np.array([(xi-stock_price + max(K-xi,0)-P_paid)*lot for xi in x_plot])
    stock_pnl = np.array([(xi-stock_price)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[
        ("Protective Put",pnl,'#157A42'),
        ("Stock Only",stock_pnl,'#B5CCFF')
    ],"Protective Put (Portfolio Insurance)",S=stock_price)
    fig.add_hline(y=-max_loss*lot,line_dash='dot',line_color='orange',
                  annotation_text=f'Max Loss ₹{max_loss*lot:.0f}')
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Max loss LIMITED to ₹{max_loss*lot:,.0f} regardless of how far Nifty falls below ₹{K:.0f}!")

elif menu == "Collar":
    st.header("🔗 Collar Strategy")
    stock_price = st.number_input("Stock Price (₹)", value=float(S))
    col1,col2 = st.columns(2)
    with col1:
        K_put = st.number_input("Put Strike K₁ (floor)", value=float(S*0.96))
        P_paid = st.number_input("Put Premium Paid (₹)", value=round(bsm(S,S*0.96,T,r,sigma,"put"),2))
    with col2:
        K_call = st.number_input("Call Strike K₂ (cap)", value=float(S*1.04))
        C_recv = st.number_input("Call Premium Received (₹)", value=round(bsm(S,S*1.04,T,r,sigma,"call"),2))
    net_cost = P_paid-C_recv; max_loss=(stock_price-K_put)+net_cost; max_gain=(K_call-stock_price)-net_cost
    strategy_card("Collar","Protect downside, sacrifice upside — near-zero cost",
        f"Long Stock + Buy {K_put:.0f}P − Sell {K_call:.0f}C",
        cr(max_gain*lot),cr(max_loss*lot),[cr(stock_price+net_cost)],
        "Long stock holder; zero-cost collar = put funded by call",
        "When expecting strong rally — upside completely capped at K₂")
    x_plot = spot_range(stock_price,pct=0.10)
    pnl = np.array([(xi-stock_price + max(K_put-xi,0)-P_paid - max(xi-K_call,0)+C_recv)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Collar",pnl,'#02B4AC')],"Collar Payoff",S=stock_price)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Net collar cost: ₹{net_cost:.2f}/unit (≈ zero-cost if designed well). "
            f"Locked range: ₹{K_put:.0f} to ₹{K_call:.0f}.")

elif menu == "Cash-Secured Put":
    st.header("💵 Cash-Secured Put")
    K = st.number_input("Put Strike K (₹) — willing to buy here", value=float(S*0.97))
    P_recv = st.number_input("Put Premium Received (₹)", value=round(bsm(S,K,T,r,sigma,"put"),2))
    effective_cost = K-P_recv
    strategy_card("Cash-Secured Put","Neutral to Bullish — willing to buy underlying at strike",
        f"Sell {K:.0f} Put @ ₹{P_recv:.2f}; hold cash = K to secure",
        cr(P_recv*lot),cr((K-P_recv)*lot),[cr(K-P_recv)],
        "When you want to buy the stock/Nifty at a LOWER price — P earned reduces cost",
        "When you do NOT want to be assigned (own the underlying)")
    x_plot = spot_range(K)
    pnl = np.array([(P_recv-max(K-xi,0))*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Cash-Secured Put",pnl,'#157A42')],"Cash-Secured Put Payoff",S=K)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"If Nifty stays above ₹{K:.0f}: keep premium ₹{P_recv*lot:,.0f}.\n"
               f"If assigned: effective buy price = ₹{effective_cost:.2f} (better than current ₹{S:.0f}!)")

elif menu == "Married Put":
    st.header("💍 Married Put")
    stock_p = st.number_input("Stock Purchase Price (₹)", value=float(S))
    K_mp = st.number_input("Put Strike K (₹)", value=float(S))
    P_mp = st.number_input("Put Premium (₹)", value=round(bsm(S,S,T,r,sigma,"put"),2))
    st.info(f"""
**Married Put = Protective Put bought AT THE SAME TIME as the stock.**

Cost basis = ₹{stock_p:.2f} + ₹{P_mp:.2f} = ₹{stock_p+P_mp:.2f}
Maximum loss = ₹{P_mp:.2f} per unit = ₹{P_mp*lot:,.0f} per lot (if Nifty falls sharply)
Break-even = ₹{stock_p+P_mp:.2f}

Often used by investors entering a new position in an uncertain market.
Think of it as 'buying stock with a seatbelt attached'.
""")
    x_plot = spot_range(stock_p,pct=0.10)
    pnl = np.array([(xi-stock_p+max(K_mp-xi,0)-P_mp)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Married Put",pnl,'#174EA6')],"Married Put Payoff",S=stock_p)
    st.plotly_chart(fig, use_container_width=True)

# ── ADVANCED ──────────────────────────────────────────────
elif menu == "Iron Condor":
    st.header("🦅 Iron Condor")
    col1,col2 = st.columns(2)
    with col1:
        K1=st.number_input("K₁ Buy Put (lowest)",value=float(S*0.94)); P1=round(bsm(S,S*0.94,T,r,sigma,"put"),2)
        K2=st.number_input("K₂ Sell Put",value=float(S*0.97)); P2=round(bsm(S,S*0.97,T,r,sigma,"put"),2)
    with col2:
        K3=st.number_input("K₃ Sell Call",value=float(S*1.03)); C3=round(bsm(S,S*1.03,T,r,sigma,"call"),2)
        K4=st.number_input("K₄ Buy Call (highest)",value=float(S*1.06)); C4=round(bsm(S,S*1.06,T,r,sigma,"call"),2)
    net_credit = (P2-P1)+(C3-C4)
    max_loss_ic = min(K2-K1, K4-K3)-net_credit
    bep_up=K3+net_credit; bep_dn=K2-net_credit
    strategy_card("Iron Condor","Strongly neutral — profit from low volatility and time decay",
        "Buy K₁P + Sell K₂P + Sell K₃C + Buy K₄C",
        cr(net_credit*lot),cr(max_loss_ic*lot),[cr(bep_dn),cr(bep_up)],
        "High VIX environment; market expected to stay in range; weekly/monthly expiry",
        "Before events — all four strikes can be tested by a single large move")
    x_plot = spot_range(S,pct=0.12)
    pnl = np.array([(net_credit-max(K2-xi,0)+max(K1-xi,0)-max(xi-K3,0)+max(xi-K4,0))*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Iron Condor",pnl,'#6A0DAD')],"Iron Condor Payoff",S=S)
    for k,lb in [(K1,'K₁'),(K2,'K₂'),(K3,'K₃'),(K4,'K₄')]:
        fig.add_vline(x=k,line_dash='dot',line_color='gray',annotation_text=lb)
    fig.add_vline(x=bep_up,line_dash='dot',line_color='red',annotation_text=f'BEP↑')
    fig.add_vline(x=bep_dn,line_dash='dot',line_color='red',annotation_text=f'BEP↓')
    st.plotly_chart(fig, use_container_width=True)
    col1,col2,col3 = st.columns(3)
    col1.metric("Net Credit Collected",cr(net_credit*lot))
    col2.metric("Upper BEP",cr(bep_up))
    col3.metric("Lower BEP",cr(bep_dn))
    st.info(f"Profit zone: ₹{bep_dn:.0f} to ₹{bep_up:.0f}. Nifty must stay within ±{pct_fmt(net_credit/S*100)} of today's level.")

elif menu == "Iron Butterfly":
    st.header("🦋 Iron Butterfly")
    col1,col2,col3 = st.columns(3)
    with col1: K1=st.number_input("K₁ Buy Put (OTM)",value=float(S*0.96)); P1=round(bsm(S,S*0.96,T,r,sigma,"put"),2)
    with col2: K2=st.number_input("K₂ ATM Strike",value=float(S))
    with col3: K3=st.number_input("K₃ Buy Call (OTM)",value=float(S*1.04)); C3=round(bsm(S,S*1.04,T,r,sigma,"call"),2)
    atm_call=round(bsm(S,K2,T,r,sigma,"call"),2); atm_put=round(bsm(S,K2,T,r,sigma,"put"),2)
    net_ib = atm_call+atm_put-P1-C3; max_loss_ib=(K2-K1)-net_ib
    strategy_card("Iron Butterfly","Strongly neutral — highest credit collected at ATM",
        "Buy K₁P + Sell ATM Call + Sell ATM Put + Buy K₃C",
        cr(net_ib*lot),cr(max_loss_ib*lot),[cr(K2-net_ib),cr(K2+net_ib)],
        "Very high VIX; expect market to pin near ATM at expiry",
        "Near major events — much higher risk than Iron Condor")
    x_plot = spot_range(K2,pct=0.10)
    pnl = np.array([(net_ib-max(K2-xi,0)+max(K1-xi,0)-max(xi-K2,0)+max(xi-K3,0))*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Iron Butterfly",pnl,'#F5A623')],"Iron Butterfly Payoff",S=K2)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Ratio Call Spread":
    st.header("📐 Ratio Call Spread")
    K1=st.number_input("K₁ (Buy 1 Call)",value=float(S))
    K2=st.number_input("K₂ (Sell 2 Calls)",value=float(S*1.02))
    C1=round(bsm(S,K1,T,r,sigma,"call"),2); C2=round(bsm(S,K2,T,r,sigma,"call"),2)
    net=C1-2*C2; max_p=(K2-K1)+net if net<0 else (K2-K1)
    strategy_card("Ratio Call Spread","Mildly bullish — profit near K₂, risky beyond",
        f"Buy 1×K₁ Call + Sell 2×K₂ Call",
        cr(max_p*lot),"Unlimited if Nifty rallies far above K₂",[cr(K1-net) if net<0 else cr(K1+abs(net))],
        "Expect moderate rally to K₂, not a large one",
        "If very bullish — naked short calls above K₂ are dangerous")
    x_plot = spot_range(K2,pct=0.08)
    pnl = np.array([(max(xi-K1,0)-2*max(xi-K2,0)-net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Ratio Call Spread",pnl,'#F5A623')],"Ratio Call Spread",S=S)
    st.plotly_chart(fig, use_container_width=True)
    st.warning("⚠️ Unlimited loss above K₂ — always monitor or add a higher strike call as a wing.")

elif menu == "Ratio Put Spread":
    st.header("📐 Ratio Put Spread")
    K1=st.number_input("K₁ (Sell 2 Puts)",value=float(S*0.98))
    K2=st.number_input("K₂ (Buy 1 Put)",value=float(S))
    P2=round(bsm(S,K2,T,r,sigma,"put"),2); P1=round(bsm(S,K1,T,r,sigma,"put"),2)
    net=P2-2*P1
    strategy_card("Ratio Put Spread","Mildly bearish — profit near K₁",
        "Buy 1×K₂ Put + Sell 2×K₁ Put",
        cr((K2-K1)*lot),"Large loss if Nifty crashes far below K₁",[cr(K2-net)],
        "Expect moderate fall to K₁, not a crash",
        "If strongly bearish or near event — risk is severe below K₁")
    x_plot = spot_range(K1,pct=0.08)
    pnl = np.array([(max(K2-xi,0)-2*max(K1-xi,0)-net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Ratio Put Spread",pnl,'#C03B3B')],"Ratio Put Spread",S=S)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Backspread (Call)":
    st.header("🚀 Call Backspread (Reverse Ratio)")
    K1=st.number_input("K₁ (Sell 1 Call)",value=float(S))
    K2=st.number_input("K₂ (Buy 2 Calls)",value=float(S*1.02))
    C1=round(bsm(S,K1,T,r,sigma,"call"),2); C2=round(bsm(S,K2,T,r,sigma,"call"),2)
    net=2*C2-C1
    strategy_card("Call Backspread","Very bullish — profits accelerate on big upward move",
        "Sell 1×K₁ Call + Buy 2×K₂ Calls",
        "Unlimited if Nifty rallies strongly",cr(net*lot) if net>0 else cr((K2-K1+net)*lot),
        [cr(K1) if net<0 else cr(K2+net)],
        "Expecting a big rally; low cost; positive gamma",
        "Neutral market — theta works against you")
    x_plot = spot_range(K2,pct=0.08)
    pnl = np.array([(-max(xi-K1,0)+2*max(xi-K2,0)-net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Call Backspread",pnl,'#157A42')],"Call Backspread Payoff",S=S)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Backspread (Put)":
    st.header("📉 Put Backspread (Reverse Ratio)")
    K1=st.number_input("K₁ (Buy 2 Puts)",value=float(S*0.98))
    K2=st.number_input("K₂ (Sell 1 Put)",value=float(S))
    P2=round(bsm(S,K2,T,r,sigma,"put"),2); P1=round(bsm(S,K1,T,r,sigma,"put"),2)
    net=2*P1-P2
    strategy_card("Put Backspread","Very bearish — profits accelerate on big crash",
        "Sell 1×K₂ Put + Buy 2×K₁ Puts",
        "Unlimited if Nifty crashes",cr(net*lot) if net>0 else "Small",
        [cr(K2) if net<0 else cr(K1-net)],
        "Expecting a sharp crash; positive gamma",
        "Neutral market — theta decay is the enemy")
    x_plot = spot_range(K1,pct=0.08)
    pnl = np.array([(-max(K2-xi,0)+2*max(K1-xi,0)-net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Put Backspread",pnl,'#C03B3B')],"Put Backspread Payoff",S=S)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Strip":
    st.header("📉 Strip (Bearish Straddle)")
    K=st.number_input("Strike K (ATM)",value=float(S))
    C=round(bsm(S,K,T,r,sigma,"call"),2); P=round(bsm(S,K,T,r,sigma,"put"),2)
    C_inp=st.number_input("Call Premium (₹)",value=C); P_inp=st.number_input("Put Premium (₹)",value=P)
    net=C_inp+2*P_inp
    strategy_card("Strip","Volatile but slightly bearish — 2× put exposure vs 1× call",
        f"Buy 1 Call + Buy 2 Puts @ K={K}",
        f"2× downside profit / Unlimited upside",cr(net*lot),
        [cr(K-(net/2)),cr(K+net)],"Expect big move with slight bearish bias","Strongly bullish markets")
    x_plot = spot_range(K,pct=0.12)
    pnl = np.array([(max(xi-K,0)+2*max(K-xi,0)-net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Strip",pnl,'#6A0DAD')],"Strip Payoff",S=K)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Strap":
    st.header("📈 Strap (Bullish Straddle)")
    K=st.number_input("Strike K (ATM)",value=float(S))
    C=round(bsm(S,K,T,r,sigma,"call"),2); P=round(bsm(S,K,T,r,sigma,"put"),2)
    C_inp=st.number_input("Call Premium (₹)",value=C); P_inp=st.number_input("Put Premium (₹)",value=P)
    net=2*C_inp+P_inp
    strategy_card("Strap","Volatile but slightly bullish — 2× call exposure vs 1× put",
        f"Buy 2 Calls + Buy 1 Put @ K={K}",
        "2× upside profit / Large downside",cr(net*lot),
        [cr(K-net),cr(K+(net/2))],"Expect big move with slight bullish bias","Strongly bearish markets")
    x_plot = spot_range(K,pct=0.12)
    pnl = np.array([(2*max(xi-K,0)+max(K-xi,0)-net)*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Strap",pnl,'#174EA6')],"Strap Payoff",S=K)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Jade Lizard":
    st.header("🦎 Jade Lizard")
    col1,col2,col3 = st.columns(3)
    with col1:
        K_put=st.number_input("Sell Put K₁",value=float(S*0.97))
        P_recv=round(bsm(S,S*0.97,T,r,sigma,"put"),2)
        P_inp=st.number_input("Put Premium Received",value=P_recv)
    with col2:
        K_c1=st.number_input("Sell Call K₂",value=float(S*1.02))
        C1_recv=round(bsm(S,S*1.02,T,r,sigma,"call"),2)
        C1_inp=st.number_input("Short Call Premium",value=C1_recv)
    with col3:
        K_c2=st.number_input("Buy Call K₃",value=float(S*1.04))
        C2_paid=round(bsm(S,S*1.04,T,r,sigma,"call"),2)
        C2_inp=st.number_input("Long Call Premium",value=C2_paid)
    net_jl=P_inp+C1_inp-C2_inp
    strategy_card("Jade Lizard","Neutral to Bullish — no upside risk at all",
        "Sell OTM Put + Sell OTM Call + Buy Higher Call (Bear Call Spread on top)",
        cr(net_jl*lot),f"On the put side only (if Nifty falls below {K_put:.0f})",[cr(K_put-net_jl)],
        "When premium collected > width of call spread — zero upside risk!",
        "Sharply bearish environment — put risk can be significant")
    x_plot = spot_range(S,pct=0.10)
    pnl = np.array([(net_jl-max(K_put-xi,0)-max(xi-K_c1,0)+max(xi-K_c2,0))*lot for xi in x_plot])
    fig = payoff_chart(x_plot,[("Jade Lizard",pnl,'#02B4AC')],"Jade Lizard Payoff",S=S)
    st.plotly_chart(fig, use_container_width=True)
    if net_jl > (K_c2-K_c1):
        st.success(f"✅ Net credit ({cr(net_jl)}) > Call spread width ({cr(K_c2-K_c1)}) → NO UPSIDE RISK!")
    else:
        st.warning(f"⚠️ Net credit ({cr(net_jl)}) < Call spread width ({cr(K_c2-K_c1)}) → Small upside risk remains.")

# ── TOOLS ─────────────────────────────────────────────────
elif menu == "Strategy Selector":
    st.header("🎯 Strategy Selector — Which Strategy Fits Your View?")
    col1,col2 = st.columns(2)
    with col1:
        view = st.selectbox("What is your market view?",
            ["Strongly Bullish (big rally expected)",
             "Mildly Bullish (moderate rise)",
             "Neutral (sideways, range-bound)",
             "Mildly Bearish (moderate fall)",
             "Strongly Bearish (big crash expected)",
             "Volatile — big move but unsure direction",
             "Very High IV — want to collect premium"])
        risk_appetite = st.radio("Risk appetite:", ["Limited (max loss = defined)", "Unlimited / High"])
        cost_pref = st.radio("Cost preference:", ["Pay premium (debit)", "Collect premium (credit)", "Either"])

    with col2:
        recs = {
            "Strongly Bullish (big rally expected)":{"Limited":["Long Call","Bull Call Spread","Call Backspread"],"Unlimited / High":["Long Futures","Long Call"]},
            "Mildly Bullish (moderate rise)":{"Limited":["Bull Call Spread","Bull Put Spread (Credit)"],"Unlimited / High":["Covered Call","Cash-Secured Put","Short Put (Naked)"]},
            "Neutral (sideways, range-bound)":{"Limited":["Iron Condor","Iron Butterfly","Short Strangle + wing"],"Unlimited / High":["Short Straddle","Short Strangle","Short Iron Butterfly"]},
            "Mildly Bearish (moderate fall)":{"Limited":["Bear Put Spread","Bear Call Spread (Credit)"],"Unlimited / High":["Short Futures","Short Call (Naked)"]},
            "Strongly Bearish (big crash expected)":{"Limited":["Long Put","Bear Put Spread","Put Backspread"],"Unlimited / High":["Short Futures","Long Put"]},
            "Volatile — big move but unsure direction":{"Limited":["Long Straddle","Long Strangle","Strip/Strap"],"Unlimited / High":["Long Straddle","Long Strangle"]},
            "Very High IV — want to collect premium":{"Limited":["Iron Condor","Iron Butterfly","Jade Lizard"],"Unlimited / High":["Short Straddle","Short Strangle"]},
        }
        strategies_for_view = recs.get(view,{}).get(risk_appetite,[])
        st.markdown("### Recommended Strategies:")
        for strat in strategies_for_view:
            st.success(f"✅ **{strat}**")

        st.markdown("### Quick Rules of Thumb:")
        st.info("""
**VIX > 20** → Prefer selling options (expensive premiums)
**VIX < 14** → Prefer buying options (cheap insurance)
**Before events** → Straddle / Strangle (buy vol)
**After events** → Short straddle / Iron Condor (sell vol crush)
**Directional + safety** → Vertical spreads (debit)
**Income + hedge** → Covered call / Protective put
""")

elif menu == "Multi-Strategy Payoff Comparator":
    st.header("📊 Multi-Strategy Payoff Comparator")
    st.markdown("Compare up to 4 strategies side-by-side on a single payoff chart.")
    S_comp = st.number_input("Spot Price", value=22000.0)
    K_comp = st.number_input("Strike (for simple strategies)", value=22000.0)
    T_comp = st.number_input("Days to Expiry", value=30) / 365
    sigma_comp = st.number_input("IV σ (%)", value=18.0) / 100
    r_comp = 0.07
    C_atm = bsm(S_comp,K_comp,T_comp,r_comp,sigma_comp,"call")
    P_atm = bsm(S_comp,K_comp,T_comp,r_comp,sigma_comp,"put")
    K_otm_c = K_comp*1.02; K_otm_p = K_comp*0.98
    C_otm = bsm(S_comp,K_otm_c,T_comp,r_comp,sigma_comp,"call")
    P_otm = bsm(S_comp,K_otm_p,T_comp,r_comp,sigma_comp,"put")

    selected = st.multiselect("Choose strategies to compare:", [
        "Long Call","Long Put","Long Straddle","Bull Call Spread",
        "Bear Put Spread","Iron Condor","Covered Call","Long Butterfly",
    ], default=["Long Call","Long Put","Long Straddle"])

    x_comp = np.linspace(K_comp*0.85, K_comp*1.15, 300)
    strategies_map = {
        "Long Call": ("Long Call",[((max(xi-K_comp,0)-C_atm)*25) for xi in x_comp],'#174EA6'),
        "Long Put": ("Long Put",[(max(K_comp-xi,0)-P_atm)*25 for xi in x_comp],'#C03B3B'),
        "Long Straddle": ("Long Straddle",[(max(xi-K_comp,0)+max(K_comp-xi,0)-C_atm-P_atm)*25 for xi in x_comp],'#6A0DAD'),
        "Bull Call Spread": ("Bull Call Spread",[(max(xi-K_comp,0)-max(xi-K_otm_c,0)-(C_atm-C_otm))*25 for xi in x_comp],'#02B4AC'),
        "Bear Put Spread": ("Bear Put Spread",[(max(K_comp-xi,0)-max(K_otm_p-xi,0)-(P_atm-P_otm))*25 for xi in x_comp],'#F5A623'),
        "Iron Condor": ("Iron Condor",[(((P_atm*0.6-P_atm*0.3)+(C_atm*0.6-C_atm*0.3))-max(K_comp*0.97-xi,0)+max(K_comp*0.94-xi,0)-max(xi-K_comp*1.03,0)+max(xi-K_comp*1.06,0))*25 for xi in x_comp],'#157A42'),
        "Covered Call": ("Covered Call",[(xi-S_comp+C_atm-max(xi-K_comp,0))*25 for xi in x_comp],'#8B4513'),
        "Long Butterfly": ("Long Butterfly",[(max(xi-K_otm_p,0)-2*max(xi-K_comp,0)+max(xi-K_otm_c,0)-(C_otm-2*C_atm+C_otm))*25 for xi in x_comp],'#FF69B4'),
    }

    fig = go.Figure()
    colors=['#174EA6','#C03B3B','#6A0DAD','#02B4AC','#F5A623','#157A42','#8B4513','#FF69B4']
    for i,strat in enumerate(selected):
        if strat in strategies_map:
            name,pnl,_ = strategies_map[strat]
            fig.add_trace(go.Scatter(x=x_comp,y=pnl,mode='lines',
                                      name=name,line=dict(color=colors[i%len(colors)],width=2)))
    fig.add_hline(y=0,line_color='black',line_width=1)
    fig.add_vline(x=S_comp,line_dash='dash',line_color='gray',annotation_text=f'Spot ₹{S_comp:.0f}')
    fig.update_layout(title="Strategy Payoff Comparison",xaxis_title="Spot at Expiry (₹)",
                      yaxis_title="P&L per lot (₹)",height=450)
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Greeks Dashboard":
    st.header("📊 Greeks Dashboard")
    S_gd=st.number_input("Spot",value=22000.0); K_gd=st.number_input("Strike",value=22000.0)
    T_gd=st.number_input("Days",value=30)/365; sigma_gd=st.number_input("σ%",value=18.0)/100
    r_gd=0.07
    g_val=greeks(S_gd,K_gd,T_gd,r_gd,sigma_gd)
    col1,col2,col3,col4=st.columns(4)
    col1.metric("Call Price",cr(bsm(S_gd,K_gd,T_gd,r_gd,sigma_gd,"call")))
    col2.metric("Put Price",cr(bsm(S_gd,K_gd,T_gd,r_gd,sigma_gd,"put")))
    col3.metric("Delta (Call)",round(g_val["dc"],4))
    col4.metric("Delta (Put)",round(g_val["dp"],4))
    col1.metric("Gamma",round(g_val["gamma"],6))
    col2.metric("Vega (per 1%σ)",cr(g_val["vega"]))
    col3.metric("Theta Call/day",cr(g_val["tc"]))
    col4.metric("Theta Put/day",cr(g_val["tp"]))
    # Delta vs spot
    spots=np.linspace(S_gd*0.85,S_gd*1.15,100)
    deltas_c=[greeks(s,K_gd,T_gd,r_gd,sigma_gd)["dc"] for s in spots]
    gammas=[greeks(s,K_gd,T_gd,r_gd,sigma_gd)["gamma"] for s in spots]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=spots,y=deltas_c,name='Call Delta',line=dict(color='#174EA6',width=2)))
    fig.add_trace(go.Scatter(x=spots,y=gammas,name='Gamma (×100)',
                              y=[g*100 for g in gammas],line=dict(color='#F5A623',width=2,dash='dash'),
                              yaxis='y2'))
    fig.add_vline(x=S_gd,line_dash='dash',line_color='gray')
    fig.update_layout(title="Delta & Gamma vs Spot",xaxis_title="Spot",yaxis_title="Delta",
                      yaxis2=dict(overlaying='y',side='right',title="Gamma ×100"))
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Implied Volatility Surface":
    st.header("🌊 Implied Volatility Surface (Skew)")
    S_ivs=st.number_input("Spot",value=22000.0)
    base_iv=st.number_input("ATM IV %",value=18.0)
    strikes=np.arange(S_ivs*0.88, S_ivs*1.12, S_ivs*0.01)
    tenors=[7,14,30,60,90]
    # Simulate vol skew + term structure
    ivs={T_d:[base_iv + 3*(S_ivs-k)/S_ivs*100 + (0.5 if T_d<=14 else 0) for k in strikes]
         for T_d in tenors}
    fig=go.Figure()
    for T_d,iv_vals in ivs.items():
        fig.add_trace(go.Scatter(x=strikes,y=iv_vals,name=f'{T_d}d',mode='lines',line=dict(width=2)))
    fig.add_vline(x=S_ivs,line_dash='dash',line_color='gray',annotation_text='ATM')
    fig.update_layout(title="Implied Volatility Skew (Put Skew Simulation)",
                      xaxis_title="Strike",yaxis_title="IV %",height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Put skew: OTM puts have higher IV than OTM calls (demand for downside protection). "
            "VIX = weighted average of near-term option IVs.")

elif menu == "Strategy P&L Simulator":
    st.header("🎲 Strategy P&L Simulator — What-If Analysis")
    S0=st.number_input("Entry Spot",value=22000.0)
    strat_sim=st.selectbox("Strategy",["Long Call","Long Put","Bull Call Spread","Long Straddle","Iron Condor"])
    T_sim=st.number_input("Days to Expiry",value=30)/365
    sigma_sim=st.number_input("IV at Entry σ%",value=18.0)/100
    r_sim=0.07; K_sim=S0
    if strat_sim=="Long Call":
        entry=bsm(S0,K_sim,T_sim,r_sim,sigma_sim,"call")
        pnl_fn=lambda s,t,iv: (bsm(s,K_sim,t,r_sim,iv,"call")-entry)*25
    elif strat_sim=="Long Put":
        entry=bsm(S0,K_sim,T_sim,r_sim,sigma_sim,"put")
        pnl_fn=lambda s,t,iv: (bsm(s,K_sim,t,r_sim,iv,"put")-entry)*25
    elif strat_sim=="Bull Call Spread":
        K2_s=S0*1.02; c1=bsm(S0,K_sim,T_sim,r_sim,sigma_sim,"call"); c2=bsm(S0,K2_s,T_sim,r_sim,sigma_sim,"call"); entry=c1-c2
        pnl_fn=lambda s,t,iv: (bsm(s,K_sim,t,r_sim,iv,"call")-bsm(s,K2_s,t,r_sim,iv,"call")-entry)*25
    elif strat_sim=="Long Straddle":
        entry=bsm(S0,K_sim,T_sim,r_sim,sigma_sim,"call")+bsm(S0,K_sim,T_sim,r_sim,sigma_sim,"put")
        pnl_fn=lambda s,t,iv: (bsm(s,K_sim,t,r_sim,iv,"call")+bsm(s,K_sim,t,r_sim,iv,"put")-entry)*25
    else:
        K3=S0*1.03; K4=S0*1.06; K1=S0*0.97; K0=S0*0.94
        p2=bsm(S0,K1,T_sim,r_sim,sigma_sim,"put"); p1=bsm(S0,K0,T_sim,r_sim,sigma_sim,"put")
        c1s=bsm(S0,K3,T_sim,r_sim,sigma_sim,"call"); c2s=bsm(S0,K4,T_sim,r_sim,sigma_sim,"call")
        entry=-(p2-p1)-(c1s-c2s)
        pnl_fn=lambda s,t,iv: (-bsm(s,K1,t,r_sim,iv,"put")+bsm(s,K0,t,r_sim,iv,"put")-bsm(s,K3,t,r_sim,iv,"call")+bsm(s,K4,t,r_sim,iv,"call")+entry)*25

    col1,col2,col3=st.columns(3)
    S_now=col1.number_input("Nifty Now",value=float(S0))
    days_elapsed=col2.number_input("Days Elapsed",value=10,min_value=0)
    iv_now=col3.number_input("Current IV %",value=18.0)/100
    T_remaining=max(T_sim-days_elapsed/365,0.001)
    pnl_now=pnl_fn(S_now,T_remaining,iv_now)
    col1.metric("Current P&L",cr(pnl_now))
    col2.metric("Days Remaining",int((T_remaining*365)))
    col3.metric("IV Change",pct_fmt((iv_now-sigma_sim)*100))

elif menu == "Quiz Engine":
    st.header("📝 Strategies Quiz Engine")
    S_q=22000; r_q=0.07; T_q=30/365; sigma_q=0.18
    questions=[
        {"q":f"Nifty=22000, IV=18%, T=30d. Bull Call Spread: Buy 22000CE @ ₹{round(bsm(S_q,22000,T_q,r_q,sigma_q,'call'),2)}, Sell 22400CE @ ₹{round(bsm(S_q,22400,T_q,r_q,sigma_q,'call'),2)}. Max profit per lot?","ans":round(((22400-22000)-(bsm(S_q,22000,T_q,r_q,sigma_q,'call')-bsm(S_q,22400,T_q,r_q,sigma_q,'call')))*25,2),"hint":"Max profit = (K2-K1-net premium) × lot size"},
        {"q":f"ATM Straddle at 22000: Call=₹{round(bsm(S_q,22000,T_q,r_q,sigma_q,'call'),2)}, Put=₹{round(bsm(S_q,22000,T_q,r_q,sigma_q,'put'),2)}. Upper BEP?","ans":round(22000+bsm(S_q,22000,T_q,r_q,sigma_q,'call')+bsm(S_q,22000,T_q,r_q,sigma_q,'put'),2),"hint":"BEP = Strike + (Call Premium + Put Premium)"},
        {"q":"Iron Condor: Sell 21400P @ ₹80, Buy 21000P @ ₹40, Sell 22600C @ ₹70, Buy 23000C @ ₹35. Net credit per unit?","ans":round(80-40+70-35,2),"hint":"Net credit = (P sold-P bought) + (C sold-C bought) = 40+35"},
    ]
    if "quiz_strat_idx" not in st.session_state: st.session_state.quiz_strat_idx=0
    if st.button("🔄 New Question"): st.session_state.quiz_strat_idx=random.randint(0,len(questions)-1); st.rerun()
    qd=questions[st.session_state.quiz_strat_idx]
    st.markdown(f"**Q:** {qd['q']}")
    ans_q=st.number_input("Your Answer (₹)",value=0.0,step=0.01)
    if st.button("Submit"):
        if abs(ans_q-qd["ans"])<max(0.5,qd["ans"]*0.02): st.success(f"✅ Correct! = ₹{qd['ans']}"); st.balloons()
        else: st.error(f"❌ Answer = ₹{qd['ans']}")
    if st.checkbox("Hint"): st.info(f"💡 {qd['hint']}")

elif menu == "Formula Cheat Sheet":
    st.header("📘 Trading Strategies — Complete Reference")
    formulas="""
DERIVATIVES TRADING STRATEGIES — COMPLETE REFERENCE
======================================================

DIRECTIONAL
──────────────────────────────────────────────────────
Long Call:          Payoff = max(ST-K,0)-C  | BEP = K+C  | Max loss = C
Short Call:         Payoff = C-max(ST-K,0)  | BEP = K+C  | Max loss = UNLIMITED
Long Put:           Payoff = max(K-ST,0)-P  | BEP = K-P  | Max loss = P
Short Put:          Payoff = P-max(K-ST,0)  | BEP = K-P  | Max loss = K-P
Long Futures:       Payoff = (ST-F)×N×Lot   | BEP = F    | Unlimited both ways
Short Futures:      Payoff = (F-ST)×N×Lot   | BEP = F    | Unlimited both ways

SPREADS (DEBIT)
──────────────────────────────────────────────────────
Bull Call Spread:   Buy K1C + Sell K2C | Net = C1-C2
  Max Profit = (K2-K1)-Net | Max Loss = Net | BEP = K1+Net
Bear Put Spread:    Buy K1P + Sell K2P (K1>K2) | Net = P1-P2
  Max Profit = (K1-K2)-Net | Max Loss = Net | BEP = K1-Net

SPREADS (CREDIT)
──────────────────────────────────────────────────────
Bull Put Spread:    Sell K1P + Buy K2P (K1>K2) | Credit = P1-P2
  Max Profit = Credit | Max Loss = (K1-K2)-Credit | BEP = K1-Credit
Bear Call Spread:   Sell K1C + Buy K2C (K2>K1) | Credit = C1-C2
  Max Profit = Credit | Max Loss = (K2-K1)-Credit | BEP = K1+Credit

VOLATILITY STRATEGIES
──────────────────────────────────────────────────────
Long Straddle:      Buy ATM Call + Buy ATM Put
  Net = C+P | BEP↑ = K+Net | BEP↓ = K-Net | Max loss = Net
Short Straddle:     Sell ATM Call + Sell ATM Put
  Max profit = C+P | Max loss = UNLIMITED | same BEPs
Long Strangle:      Buy OTM Call (K2) + Buy OTM Put (K1)
  BEP↑ = K2+Net | BEP↓ = K1-Net | Max loss = Net
Short Strangle:     Sell OTM Call + Sell OTM Put (wider zone)
Long Butterfly:     Buy K1C + Sell 2×K2C + Buy K3C
  Net = C1-2C2+C3 | Max Profit = (K2-K1)-Net at K2
Strip:              Buy 1 Call + Buy 2 Puts (bearish bias vol)
Strap:              Buy 2 Calls + Buy 1 Put (bullish bias vol)

INCOME & HEDGING
──────────────────────────────────────────────────────
Covered Call:       Long Stock + Sell Call
  Max Profit = (K-S0)+C | BEP = S0-C
Protective Put:     Long Stock + Buy Put
  Max Loss = (S0-K)+P (LIMITED!) | BEP = S0+P
Collar:             Long Stock + Buy Put(K1) + Sell Call(K2)
  Range locked K1 to K2 | Near-zero net cost possible
Cash-Secured Put:   Sell Put; hold cash=K to secure assignment
  Effective buy price = K-premium if assigned

ADVANCED
──────────────────────────────────────────────────────
Iron Condor:        Buy K1P+Sell K2P+Sell K3C+Buy K4C
  Max Profit = net credit | Max Loss = width-credit
Iron Butterfly:     Buy OTM Put+Sell ATM Call+Sell ATM Put+Buy OTM Call
  Highest credit; narrow profit zone at ATM
Ratio Spread:       Buy 1 + Sell 2 at different strikes (extra upside short)
Backspread:         Sell 1 + Buy 2 (unlimited profit on big move)
Calendar:           Sell near + Buy far (same strike): profit from theta diff
Diagonal:           Sell near (OTM) + Buy far (ITM or diff strike)
Jade Lizard:        Short Put + Short Call Spread (no upside risk)

KEY RULES
──────────────────────────────────────────────────────
VIX > 20 → Sell premium (Iron Condor, Straddle, Strangle)
VIX < 14 → Buy premium (Straddle, Calls, Puts)
Before events → Buy volatility (Long Straddle / Strangle)
After events → Sell volatility (Short Straddle, Iron Condor)
Directional + limited risk → Vertical spreads (Bull/Bear)
Income generation → Covered Call, Cash-Secured Put
Portfolio crash hedge → Protective Put, Collar
======================================================
"""
    st.text_area("Reference",formulas,height=700)
    st.download_button("📥 Download",data=formulas,file_name="Trading_Strategies_Reference.txt")
