import yfinance as yf
import numpy as np
import pandas as pd
import sys
import os
import requests
from datetime import datetime
import pytz
import time

# --- [1. 전략 파라미터 설정] ---
TICKERS = ['QQQ', 'TLT', 'GLD']
BASE_WEIGHTS = {
    'QQQ': 0.45,
    'TLT': 0.35,
    'GLD': 0.20
}
N_BAND = 0.03 # 3% 이격도
MA_WINDOWS = [20, 120, 200]
SCALAR_MAP = {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0} # 시나리오 A

# 텔레그램 Secrets
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_TO = os.environ.get('TELEGRAM_TO')

# --- [2. 텔레그램 전송 함수] ---
def send_telegram_message(token, chat_id, message, parse_mode='Markdown'):
    if not token or not chat_id:
        print("텔레그램 TOKEN 또는 CHAT_ID가 설정되지 않았습니다.", file=sys.stderr)
        return False
        
    # [수정됨] URL에 마크다운 서식이 들어가지 않도록 주의
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    # 메시지가 길어질 수 있으므로 타임아웃을 넉넉히 설정
    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': parse_mode}
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        print("텔레그램 메시지 전송 성공.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"텔레그램 전송 실패: {e}", file=sys.stderr)
        return False

# --- [3. 일일 신호 계산 및 리포트 생성] ---
def get_daily_signals_and_report():
    
    print("... 최신 시장 데이터 다운로드 중 ...")
    data_full = yf.download(TICKERS, period="400d", progress=False)
    
    if data_full.empty:
        raise ValueError("데이터 다운로드에 실패했습니다.")
    
    prices_df = data_full['Close'].ffill()
    
    # --- 이격도(Hysteresis) 상태 계산 ---
    
    ma_lines = {}
    upper_bands = {}
    lower_bands = {}
    
    for ticker in TICKERS:
        for window in MA_WINDOWS:
            ma_key = f"{ticker}_{window}"
            ma_lines[ma_key] = prices_df[ticker].rolling(window=window).mean()
            upper_bands[ma_key] = ma_lines[ma_key] * (1.0 + N_BAND)
            lower_bands[ma_key] = ma_lines[ma_key] * (1.0 - N_BAND)

    yesterday_ma_states = {f"{ticker}_{window}": 0.0 for ticker in TICKERS for window in MA_WINDOWS}
    
    today_scalars = pd.Series(0.0, index=TICKERS)
    yesterday_scalars = pd.Series(0.0, index=TICKERS)
    
    today_ma_states_dict = yesterday_ma_states.copy()
    yesterday_ma_states_dict = yesterday_ma_states.copy()

    start_index = max(MA_WINDOWS) - 1 
    
    for i in range(start_index, len(prices_df)):
        
        today_scores = pd.Series(0, index=TICKERS)
        current_ma_states = {}
        
        for ticker in TICKERS:
            score = 0
            for window in MA_WINDOWS:
                ma_key = f"{ticker}_{window}"
                yesterday_state = yesterday_ma_states[ma_key]
                
                price = prices_df[ticker].iloc[i]
                upper = upper_bands[ma_key].iloc[i]
                lower = lower_bands[ma_key].iloc[i]
                
                if pd.isna(upper): new_state = 0.0
                elif yesterday_state == 1.0: 
                    new_state = 1.0 if price >= lower else 0.0
                else: 
                    new_state = 1.0 if price > upper else 0.0
                
                current_ma_states[ma_key] = new_state
                score += new_state
            
            today_scores[ticker] = score
        
        if i == len(prices_df) - 2:
            yesterday_scalars = today_scores.map(SCALAR_MAP)
            yesterday_ma_states_dict = current_ma_states
        if i == len(prices_df) - 1:
            today_scalars = today_scores.map(SCALAR_MAP)
            today_ma_states_dict = current_ma_states
        
        yesterday_ma_states = current_ma_states

    # --- 비중 계산 ---
    today_weights = (today_scalars * pd.Series(BASE_WEIGHTS)).to_dict()
    yesterday_weights = (yesterday_scalars * pd.Series(BASE_WEIGHTS)).to_dict()
    
    today_total_cash = 1.0 - sum(today_weights.values())
    yesterday_total_cash = 1.0 - sum(yesterday_weights.values())
    
    is_rebalancing_needed = not (today_scalars.equals(yesterday_scalars))
    
    # --- [리포트 작성 통합] ---
    
    yesterday = prices_df.index[-1]
    kst = pytz.timezone('Asia/Seoul')
    if yesterday.tzinfo is None:
        yesterday_kst = kst.localize(yesterday)
    else:
        yesterday_kst = yesterday.astimezone(kst)
    
    report = []
    report.append(f"🔔 **Independent-Hysteresis-TAA**")
    report.append(f"({yesterday_kst.strftime('%Y-%m-%d %A')} 마감 기준)")

    # [1] 신호
    if is_rebalancing_needed:
        report.append("\n🔼 **리밸런싱: 매매 필요**")
        report.append("(목표 비중이 변경되었습니다)")
    else:
        report.append("\n🟢 **리밸런싱: 매매 불필요**")
        report.append("(비중 유지)")
    
    report.append("\n" + "-"*20)

    # [2] 목표 비중
    report.append("💰 **[1] 오늘 목표 비중**")
    
    for ticker in TICKERS:
        emoji = "🎯" if today_weights[ticker] != yesterday_weights[ticker] else "*"
        report.append(f"{emoji} {ticker}: {today_weights[ticker]:.1%}")
    
    cash_emoji = "🎯" if abs(today_total_cash - yesterday_total_cash) > 0.0001 else "*"
    report.append(f"{cash_emoji} 현금 (Cash): {today_total_cash:.1%}")
    
    report.append("\n" + "-"*20)

    # [3] 비중 변경 상세
    report.append("📊 **[2] 비중 변경 상세**")
    
    def format_change_row(name, yesterday, today):
        delta = today - yesterday
        if abs(delta) < 0.0001:
            change_str = "(유지)"
        else:
            emoji = "🔼" if delta > 0 else "🔽"
            change_str = f"{emoji} {delta:+.1%}"
        
        return f"{name}: {yesterday:.1%} → {today:.1%} | {change_str}"

    for ticker in TICKERS:
        report.append(format_change_row(ticker, yesterday_weights[ticker], today_weights[ticker]))
    
    report.append(format_change_row('현금', yesterday_total_cash, today_total_cash))
    
    report.append("\n" + "-"*20)
    
    # [4] 시장 현황
    report.append("📈 **[3] 전일 시장 현황**")
    
    today_prices = prices_df.iloc[-1]
    price_change = prices_df.pct_change().iloc[-1]
    
    for ticker in TICKERS:
        emoji = "🔴" if price_change[ticker] >= 0 else "🔵"
        report.append(f"{emoji} {ticker}: ${today_prices[ticker]:.2f} ({price_change[ticker]:+.1%})")
    
    report.append("\n" + "-"*20)

    # [5] MA 상세
    report.append("🔍 **[4] MA 신호 상세**")
    report.append(f"(이격도 +/- {N_BAND:.1%} 룰)")
    
    for ticker in TICKERS:
        score = int(today_scalars[ticker] * 4 / (4/3))
        status_emoji = "🟢ON" if score > 0 else "🔴OFF"
        report.append(f"\n**{ticker} ({score}/3 {status_emoji})**")
        
        for window in MA_WINDOWS:
            ma_key = f"{ticker}_{window}"
            today_state = today_ma_states_dict[ma_key]
            yesterday_state = yesterday_ma_states_dict[ma_key]
            
            state_emoji = "ON" if today_state == 1.0 else "OFF"
            
            if today_state > yesterday_state: state_change = "[신규 ON]"
            elif today_state < yesterday_state: state_change = "[신규 OFF]"
            else: state_change = ""
            
            t_price = today_prices[ticker]
            ma_val = ma_lines[ma_key].iloc[-1]
            disparity = (t_price / ma_val) - 1.0
            
            report.append(f"- {window}일: {state_emoji} ({disparity:.1%}) {state_change}")

    return "\n".join(report)

# --- [4. 메인 실행] ---
if __name__ == "__main__":
    try:
        # 1. 리포트 생성
        full_report = get_daily_signals_and_report()
        print(full_report)
        
        # 2. 텔레그램 전송
        if send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_TO, full_report, parse_mode='Markdown'):
            print("전송 완료.")
        else:
            raise Exception("전송 실패")
        
    except Exception as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)
