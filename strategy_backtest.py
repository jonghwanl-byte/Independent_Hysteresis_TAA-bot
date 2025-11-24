import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams['axes.unicode_minus'] = False

# --- 설정 ---
TICKERS = ['QQQ', 'TLT', 'GLD']
BASE_WEIGHTS = {'QQQ': 0.45, 'TLT': 0.35, 'GLD': 0.20}
N_BAND = 0.03
MA_WINDOWS = [20, 120, 200]
SCALAR_MAP = {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0}
START_DATE = '2004-11-18'

# --- 함수 ---
def get_mdd(cum_returns):
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

def get_cagr(cum_returns):
    total_ret = cum_returns.iloc[-1]
    days = (cum_returns.index[-1] - cum_returns.index[0]).days
    years = days / 365.25
    if years <= 0: return 0
    return (total_ret) ** (1 / years) - 1

def get_sharpe(returns):
    if returns.std() == 0: return 0
    return (returns.mean() * 252) / (returns.std() * np.sqrt(252))

def run_strategy(base_weights, returns_df, prices_df):
    
    ma_lines = {}
    upper_bands = {}
    lower_bands = {}
    
    for ticker in TICKERS:
        for w in MA_WINDOWS:
            ma = prices_df[ticker].rolling(window=w).mean()
            ma_lines[f"{ticker}_{w}"] = ma
            upper_bands[f"{ticker}_{w}"] = ma * (1 + N_BAND)
            lower_bands[f"{ticker}_{w}"] = ma * (1 - N_BAND)
            
    portfolio_returns = []
    states = {f"{t}_{w}": 0.0 for t in TICKERS for w in MA_WINDOWS}
    
    start_idx = max(MA_WINDOWS)
    dates = returns_df.index[start_idx:]
    
    for i in range(start_idx, len(returns_df)):
        daily_ret = 0
        for ticker in TICKERS:
            score = 0
            for w in MA_WINDOWS:
                key = f"{ticker}_{w}"
                price = prices_df[ticker].iloc[i]
                upper = upper_bands[key].iloc[i]
                lower = lower_bands[key].iloc[i]
                prev_state = states[key]
                
                if pd.isna(upper): new_state = 0.0
                elif prev_state == 1.0: new_state = 1.0 if price >= lower else 0.0
                else: new_state = 1.0 if price > upper else 0.0
                
                states[key] = new_state
                score += new_state
            
            scalar = SCALAR_MAP.get(score, 0.0)
            daily_ret += base_weights[ticker] * scalar * returns_df[ticker].iloc[i]
            
        portfolio_returns.append(daily_ret)
        
    return pd.Series(portfolio_returns, index=dates)

# --- 실행 ---
try:
    print(f"데이터 다운로드 중... ({', '.join(TICKERS)})")
    data = yf.download(TICKERS, start=START_DATE, progress=False)
    prices = data['Close'].ffill()
    returns = prices.pct_change().dropna()
    prices = prices.loc[returns.index]
    
    print("전략 계산 중...")
    final_ret = run_strategy(BASE_WEIGHTS, returns, prices)
    final_cum = (1 + final_ret).cumprod()
    
    cagr = get_cagr(final_cum)
    mdd = get_mdd(final_cum)
    sharpe = get_sharpe(final_ret)
    
    print("\n" + "="*50)
    print(f"📊 전략 성과 리포트 (Independent MA)")
    print("-" * 50)
    print(f"CAGR : {cagr:.2%}")
    print(f"MDD  : {mdd:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print("="*50)
    
    plt.figure(figsize=(12, 6))
    plt.plot(final_cum, label='Strategy Portfolio')
    plt.title('Cumulative Returns')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()

except Exception as e:
    print(f"오류 발생: {e}")
