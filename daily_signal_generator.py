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
ASSET_NAMES = [
    '한국 주식', '중국 주식', '인도 주식', 
    '채권 10년', '채권 30년'
]
TICKER_MAP = {
    '한국 주식': '102110.KS',
    '중국 주식': '283580.KS',
    '인도 주식': '453810.KS',
    '채권 10년': '148070.KS',
    '채권 30년': '385560.KS'
}
TICKER_LIST = list(TICKER_MAP.values())

BASE_WEIGHTS = {name: 0.20 for name in ASSET_NAMES} 
MA_WINDOWS = [20, 120, 200]
N_BAND = 0.03 
SCALAR_MAP = {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0} 

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_TO = os.environ.get('TELEGRAM_TO')

# --- [2. 텔레그램 전송 함수] ---
def send_telegram_message(token, chat_id, message, parse_mode='Markdown'):
    if not token or not chat_id:
        print("텔레그램 토큰 오류", file=sys.stderr)
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': parse_mode}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("전송 성공")
        return True
    except requests.exceptions.RequestException as e:
        print(f"전송 실패: {e}", file=sys.stderr)
        return False

# --- [3. 신호 계산 및 리포트 생성] ---
def get_daily_signals_and_report():
    
    print("... 데이터 다운로드 중 ...")
    data_full = yf.download(TICKER_LIST, period="400d", progress=False)
    
    if data_full.empty:
        raise ValueError("데이터 다운로드 실패")
    
    all_prices_df_raw = data_full['Close'].ffill()
    all_prices_df = all_prices_df_raw.rename(columns={v: k for k, v in TICKER_MAP.items()})
    
    # MA 및 이격도 계산
    ma_lines = {}
    upper_bands = {}
    lower_bands = {}
    for name in ASSET_NAMES:
        for window in MA_WINDOWS:
            ma_key = f"{name}_{window}"
            ma_lines[ma_key] = all_prices_df[name].rolling(window=window).mean()
            upper_bands[ma_key] = ma_lines[ma_key] * (1.0 + N_BAND)
            lower_bands[ma_key] = ma_lines[ma_key] * (1.0 - N_BAND)

    yesterday_ma_states = {f"{name}_{window}": 0.0 for name in ASSET_NAMES for window in MA_WINDOWS}
    
    today_scalars = pd.Series(0.0, index=ASSET_NAMES)
    yesterday_scalars = pd.Series(0.0, index=ASSET_NAMES)
    
    today_ma_states_dict = yesterday_ma_states.copy()
    yesterday_ma_states_dict = yesterday_ma_states.copy()

    start_index = max(MA_WINDOWS) - 1 
    
    for i in range(start_index, len(all_prices_df)):
        
        today_scores = pd.Series(0, index=ASSET_NAMES)
        current_ma_states = {}
        
        for name in ASSET_NAMES:
            score = 0
            for window in MA_WINDOWS:
                ma_key = f"{name}_{window}"
                yesterday_state = yesterday_ma_states[ma_key]
                
                price = all_prices_df[name].iloc[i]
                upper = upper_bands[ma_key].iloc[i]
                lower = lower_bands[ma_key].iloc[i]
                
                if pd.isna(upper): new_state = 0.0
                elif yesterday_state == 1.0: 
                    new_state = 1.0 if price >= lower else 0.0
                else: 
                    new_state = 1.0 if price > upper else 0.0
                
                current_ma_states[ma_key] = new_state
                score += new_state
            
            today_scores[name] = score
        
        if i == len(all_prices_df) - 2:
            yesterday_scalars = today_scores.map(SCALAR_MAP)
            yesterday_ma_states_dict = current_ma_states
        if i == len(all_prices_df) - 1:
            today_scalars = today_scores.map(SCALAR_MAP)
            today_ma_states_dict = current_ma_states
        
        yesterday_ma_states = current_ma_states

    # 비중 계산
    today_weights = (today_scalars * pd.Series(BASE_WEIGHTS)).to_dict()
    yesterday_weights = (yesterday_scalars * pd.Series(BASE_WEIGHTS)).to_dict()
    
    today_total_cash = 1.0 - sum(today_weights.values())
    yesterday_total_cash = 1.0 - sum(yesterday_weights.values())
    
    is_rebalancing_needed = not (today_scalars.equals(yesterday_scalars))
    
    # --- 리포트 작성 ---
    yesterday = all_prices_df.index[-1]
    kst = pytz.timezone('Asia/Seoul')
    if yesterday.tzinfo is None:
        yesterday_kst = kst.localize(yesterday)
    else:
        yesterday_kst = yesterday.astimezone(kst)
    
    report = []
    report.append(f"🔔 **TAA 5-Asset Bot**")
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
    for name in ASSET_NAMES:
        emoji = "🎯" if today_weights[name] != yesterday_weights[name] else "*"
        report.append(f"{emoji} {name}: {today_weights[name]:.1%}")
    
    cash_emoji = "🎯" if abs(today_total_cash - yesterday_total_cash) > 0.0001 else "*"
    report.append(f"{cash_emoji} 현금 (Cash): {today_total_cash:.1%}")
    
    report.append("\n" + "-"*20)

    # [3] 비중 변경 상세 (박스 제거)
    report.append("📊 **[2] 비중 변경 상세**")
    # 박스(```) 제거됨
    
    def format_change_row(name, yesterday, today):
        delta = today - yesterday
        if abs(delta) < 0.0001:
            change_str = "(유지)"
        else:
            emoji = "🔼" if delta > 0 else "🔽"
            change_str = f"{emoji} {delta:+.1%}"
        
        # 텍스트 정렬 (박스가 없으므로 완벽한 정렬은 어려울 수 있음)
        return f"{name}: {yesterday:.1%} → {today:.1%} | {change_str}"

    for name in ASSET_NAMES:
        report.append(format_change_row(name, yesterday_weights[name], today_weights[name]))
    
    report.append(format_change_row('현금', yesterday_total_cash, today_total_cash))
    
    report.append("\n" + "-"*20)
    
    # [4] 시장 현황
    report.append("📈 **[3] 전일 시장 현황**")
    today_prices = all_prices_df.iloc[-1]
    price_change = all_prices_df.pct_change().iloc[-1]
    
    for name in ASSET_NAMES:
        emoji = "🔴" if price_change[name] >= 0 else "🔵"
        report.append(f"{emoji} {name}: {today_prices[name]:,.0f} ({price_change[name]:+.1%})")
    
    report.append("\n" + "-"*20)

    # [5] MA 상세
    report.append("🔍 **[4] MA 신호 상세**")
    report.append(f"(이격도 +/- {N_BAND:.1%} 룰)")
    
    for name in ASSET_NAMES:
        score = int(today_scalars[name] * 4 / (4/3))
        status_emoji = "🟢ON" if score > 0 else "🔴OFF"
        report.append(f"\n**{name} ({score}/3 {status_emoji})**")
        
        for window in MA_WINDOWS:
            ma_key = f"{name}_{window}"
            today_state = today_ma_states_dict[ma_key]
            yesterday_state = yesterday_ma_states_dict[ma_key]
            
            state_emoji = "ON" if today_state == 1.0 else "OFF"
            
            if today_state > yesterday_state: state_change = "[신규 ON]"
            elif today_state < yesterday_state: state_change = "[신규 OFF]"
            else: state_change = ""
            
            t_price = today_prices[name]
            ma_val = ma_lines[ma_key].iloc[-1]
            disparity = (t_price / ma_val) - 1.0
            
            report.append(f"- {window}일: {state_emoji} ({disparity:+.1%}) {state_change}")

    # [수정] 하나의 긴 문자열로 반환
    return "\n".join(report)

# --- [4. 메인 실행] ---
if __name__ == "__main__":
    try:
        # 1. 리포트 생성 (하나의 문자열)
        full_report = get_daily_signals_and_report()
        print(full_report)
        
        # 2. 텔레그램 전송 (한 번만 전송)
        if send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_TO, full_report, parse_mode='Markdown'):
            print("전송 완료.")
        else:
            raise Exception("전송 실패")
        
    except Exception as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)
