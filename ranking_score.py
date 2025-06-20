import pandas as pd
import requests
from bs4 import BeautifulSoup
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from io import StringIO
import os
import concurrent.futures

import time

# safe_float 함수는 이 파일에서 사용되므로 여기에 정의
def safe_float(value):
    try:
        # 콤마 제거 후 float 변환
        return float(str(value).replace(",", ""))
    except (ValueError, AttributeError):
        return np.nan

# ✅ 종목 코드 매핑용 함수
def fetch_code_map():
    url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
    try:
        df = pd.read_html(url, encoding='euc-kr')[0]
        df = df[['회사명', '종목코드']]
        df['종목코드'] = df['종목코드'].apply(lambda x: f"{x:06d}")
        return dict(zip(df['회사명'], df['종목코드']))
    except Exception as e:
        print(f"⚠️ KRX 종목 코드 매핑 실패: {e}")
        return {}

# ✅ 기술적 지표용 실제 종가 시퀀스, 거래량 가져오기
def fetch_real_close_prices(code, days=120):
    sise_url = f"https://finance.naver.com/item/sise_day.nhn?code={code}"

    # ⭐ 최신 User-Agent로 업데이트
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    
    close_prices = pd.Series([], dtype=float)
    volumes = pd.Series([], dtype=float)

    dfs_sise = []
    for page in range(1, 10): # 최대 9페이지까지만 시도 (대략 1년치 데이터)
        pg_url = f"{sise_url}&page={page}"
        try:
            res = requests.get(pg_url, headers=headers, timeout=10)
            if res.status_code != 200:
                break
            
            # ⭐ HTML 내용을 StringIO로 전달하기 전에 유효성 검사
            if "캡차" in res.text or "차단" in res.text: # 간단한 캡차/차단 메시지 확인
                print(f"⚠️ 경고: {code} 종목 시세 크롤링 중 캡차/IP 차단 감지. 페이지: {page}")
                break

            df = pd.read_html(StringIO(res.text))[0]
            if df.empty or '날짜' not in df.columns: # '날짜' 컬럼이 없으면 유효한 테이블이 아님
                break
            
            # 필요한 컬럼만 선택하여 데이터 정리
            df = df[['날짜', '종가', '거래량']].dropna()
            df['날짜'] = pd.to_datetime(df['날짜'])
            df = df.sort_values(by='날짜').reset_index(drop=True)
            dfs_sise.append(df)
            
            time.sleep(0.1) # 요청 간 짧은 딜레이 추가
        except (requests.exceptions.RequestException, ValueError, IndexError) as e:
            # print(f"⚠️ {code} 종목 시세 크롤링 오류 (페이지 {page}): {e}") # 디버깅용
            break # 오류 발생 시 해당 종목의 시세 크롤링 중단

    if dfs_sise:
        df_sise = pd.concat(dfs_sise, ignore_index=True)
        # 중복된 날짜 제거 (페이지가 겹쳐서 같은 날짜가 여러번 나올 수 있음)
        df_sise = df_sise.drop_duplicates(subset=['날짜']).sort_values(by='날짜')
        close_prices = df_sise['종가'].astype(float).tail(days)
        volumes = df_sise['거래량'].astype(float).tail(days)
    
    return close_prices, volumes


# ✅ 시장별 주식 목록 및 기본 정보 수집 (외국인비율 컬럼 추가)
def fetch_stock_list(market='kospi', pages=10):
    base_url = {
        'kospi': 'https://finance.naver.com/sise/sise_market_sum.nhn?sosok=0&page=',
        'kosdaq': 'https://finance.naver.com/sise/sise_market_sum.nhn?sosok=1&page='
    }[market]

    # ⭐ 최신 User-Agent로 업데이트
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    dfs = []
    required_cols = ['종목명', '현재가', 'PER', 'ROE'] # 필수 컬럼
    foreign_rate_col_candidates = ['외국인비율', '외국인비율(%)', '외국인보유율'] # 외국인 비율 컬럼 후보들

    for page in range(1, pages + 1):
        url = base_url + str(page)
        print(f"--- 🌐 {market.upper()} 페이지 {page} 크롤링 시도 중 ---")
        try:
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code != 200:
                print(f"⚠️ 페이지 {page} 접속 실패: Status Code {res.status_code}")
                continue
            
            # ⭐ 캡차/차단 감지
            if "캡차" in res.text or "차단" in res.text or "bot_block" in res.text:
                print(f"❌ 경고: {market.upper()} 시장 크롤링 중 캡차/IP 차단 감지. 페이지: {page}. 크롤링 중단.")
                break # 차단 감지 시 루프 종료

            soup = BeautifulSoup(res.text, "html.parser")
            table_selector = "table.type_2"
            table = soup.select_one(table_selector) # CSS 셀렉터로 테이블 요소 직접 찾기

            if not table: # 테이블 요소 자체를 찾지 못했다면
                print(f"⚠️ 페이지 {page}에서 '{table_selector}' 테이블을 찾을 수 없습니다. (HTML 구조 변경 또는 빈 페이지)")
                continue # 다음 페이지로 넘어감

            # ⭐ BeautifulSoup로 찾은 테이블의 HTML을 StringIO로 전달
            # pd.read_html은 테이블 태그 내부의 내용을 읽도록 수정
            tables_from_html = pd.read_html(StringIO(str(table))) # 찾은 테이블만 파싱하도록 변경
            
            df_page = pd.DataFrame()
            found_target_table = False

            for temp_df in tables_from_html:
                # pandas_html_parser가 가끔 테이블 외의 다른 것을 읽어올 수 있어 컬럼이 없는 경우 필터링 강화
                if temp_df.empty or len(temp_df.columns) == 0:
                    continue

                # '종목명' 컬럼이 없으면 유효한 주식 목록 테이블이 아니라고 판단
                if '종목명' not in temp_df.columns:
                    continue

                has_all_required_base_cols = all(col in temp_df.columns for col in required_cols)
                has_foreign_rate_col = any(col in temp_df.columns for col in foreign_rate_col_candidates)
                
                if has_all_required_base_cols and has_foreign_rate_col:
                    df_page = temp_df
                    found_target_table = True
                    break
            
            if not found_target_table:
                print(f"⚠️ 페이지 {page}에서 필수 컬럼을 포함하는 테이블을 찾을 수 없습니다. (데이터 부족 또는 구조 미스매치)")
                continue 

            foreign_rate_col_name_found = None
            for col_candidate in foreign_rate_col_candidates:
                if col_candidate in df_page.columns:
                    foreign_rate_col_name_found = col_candidate
                    break
            
            if not foreign_rate_col_name_found:
                # 이 경우는 상위 if not found_target_table 에서 이미 걸러질 가능성이 높음
                print(f"⚠️ 페이지 {page}에서 외국인비율 컬럼을 찾을 수 없습니다.")
                continue 

            df_page = df_page.rename(columns={foreign_rate_col_name_found: '외국인비율'})
            
            final_cols = required_cols + ['외국인비율']

            if not all(col in df_page.columns for col in final_cols):
                missing_cols = [col for col in final_cols if col not in df_page.columns]
                print(f"⚠️ 페이지 {page} - 최종 컬럼 선택 후 누락된 컬럼: {missing_cols}. 건너뜁니다.")
                continue

            df_page = df_page[final_cols].copy() 

            df_page['현재가'] = df_page['현재가'].apply(safe_float)
            df_page['PER'] = df_page['PER'].apply(safe_float)
            df_page['ROE'] = df_page['ROE'].apply(safe_float)
            df_page['외국인비율'] = df_page['외국인비율'].apply(safe_float)

            df_page = df_page.dropna(subset=final_cols) # 필수 컬럼 중 NaN 있는 행 제거
            
            df_page['현재가'] = df_page['현재가'].astype(int)

            if df_page.empty:
                print(f"⚠️ 페이지 {page} - NaN 값 제거 후 데이터가 없습니다. 건너뜁니다.")
                continue
            
            dfs.append(df_page)
            print(f"✅ 페이지 {page}에서 {len(df_page)}개 종목 수집 완료.")
            
            time.sleep(0.5) # 페이지 요청 간 딜레이 추가 (IP 차단 방지)

        except (requests.exceptions.RequestException, ValueError, IndexError, KeyError, TypeError) as e:
            print(f"❌ 시장 목록 크롤링 오류 (페이지 {page}, 상세: {e})")
            continue 

    if not dfs:
        print("❌ 시장 목록을 가져오는 데 실패했습니다. 빈 DataFrame 반환.")
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna(subset=required_cols + ['외국인비율']) 
    
    return full_df.reset_index(drop=True)


# 각 종목을 처리하는 헬퍼 함수
def _process_single_stock(stock_info, code_map):
    name = stock_info['종목명']
    code = code_map.get(name)
    if not code:
        return None

    try:
        close_prices, volumes = fetch_real_close_prices(code) 
        
        current_foreign_rate = stock_info.get('외국인비율', np.nan) 
        
        if len(close_prices) < 60 or len(volumes) < 60: 
            return None

        rsi_series = ta.rsi(close_prices, length=14)
        macd_df = ta.macd(close_prices)
        bb_df = ta.bbands(close_prices)

        if rsi_series.empty or macd_df.empty or bb_df.empty:
             return None

        current_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2] if len(rsi_series) >= 2 else np.nan
        macd_line = macd_df["MACD_12_26_9"].iloc[-1]
        signal_line = macd_df["MACDs_12_26_9"].iloc[-1]
        prev_macd_line = macd_df["MACD_12_26_9"].iloc[-2] if len(macd_df) >= 2 else np.nan
        prev_signal_line = macd_df["MACDs_12_26_9"].iloc[-2] if len(macd_df) >= 2 else np.nan
        percent_b = bb_df["BBP_5_2.0"].iloc[-1]
        bbl = bb_df["BBL_5_2.0"].iloc[-1]
        bbm = bb_df["BBM_5_2.0"].iloc[-1] # 볼린저밴드 중간선 (20일 이평선)
        current_close_price = close_prices.iloc[-1]
        current_volume = volumes.iloc[-1]
        
        avg_volume_20_days = volumes.rolling(window=20).mean().iloc[-1] if len(volumes) >= 20 else np.nan

        foreign_ownership_change = np.nan 
        
        is_foreign_inflow = True if pd.notna(current_foreign_rate) and current_foreign_rate >= 5 else False 

        per = stock_info["PER"] if pd.notna(stock_info["PER"]) and stock_info["PER"] > 0 else 1000
        roe = stock_info["ROE"] if pd.notna(stock_info["ROE"]) else 0

        # --- ⭐ 급등_점수 10점 만점 세분화 로직 시작 ⭐ ---
        surge_score = 0
        temp_surge_signals = [] # 중복 처리를 위해 임시 리스트 사용

        # 1. RSI 점수 (최대 2.5점)
        is_rsi_recovery = False
        if pd.notna(prev_rsi) and prev_rsi < 30 and current_rsi >= 30:
            is_rsi_recovery = True
            surge_score += 1.0 # RSI 회복 기본 점수
            
            if current_rsi >= 50:
                surge_score += 1.0 # RSI 50+ 돌파
                temp_surge_signals.append("RSI 50+ 돌파")
            
            if prev_rsi < 20: # 더 깊은 과매도에서 회복
                surge_score += 0.5
                temp_surge_signals.append("RSI 깊은 과매도 회복")
            
            if not ("RSI 50+ 돌파" in temp_surge_signals or "RSI 깊은 과매도 회복" in temp_surge_signals):
                temp_surge_signals.append("RSI 회복")


        # 2. MACD 점수 (최대 2.5점)
        is_macd_golden_cross = False
        if pd.notna(prev_macd_line) and pd.notna(prev_signal_line) and \
           macd_line > signal_line and prev_macd_line <= prev_signal_line:
            is_macd_golden_cross = True
            surge_score += 1.0 # MACD 골든 크로스 기본 점수
            
            if macd_line > 0 and prev_macd_line <= 0: # MACD 0선 돌파
                surge_score += 1.0
                temp_surge_signals.append("MACD GC & 0선 돌파")
            
            macd_diff = macd_line - signal_line
            if macd_diff > 0.5: # MACD 이격 확대 (상승 강도)
                surge_score += 0.5
                if "MACD GC & 0선 돌파" not in temp_surge_signals:
                    temp_surge_signals.append("MACD 이격 확대")
            
            if not ("MACD GC & 0선 돌파" in temp_surge_signals or "MACD 이격 확대" in temp_surge_signals):
                temp_surge_signals.append("MACD GC")


        # 3. 볼린저밴드 점수 (최대 2점)
        is_bb_lower_band_recovery = False
        if len(close_prices) >= 10 and not bb_df["BBL_5_2.0"].iloc[-10:].isnull().any():
            was_below_bbl = any(close_prices.iloc[-10:-1] < bb_df["BBL_5_2.0"].iloc[-10:-1])
            is_above_bbl_now = current_close_price > bbl
            if was_below_bbl and is_above_bbl_now:
                is_bb_lower_band_recovery = True
                surge_score += 1.0 # BB 하단 회복 기본 점수

                if current_close_price > bbm and close_prices.iloc[-2] <= bbm: # BB 중간선 돌파
                    surge_score += 0.5
                    temp_surge_signals.append("BB 하단 & 중간선 돌파")
                
                bb_width = bb_df["BBU_5_2.0"].iloc[-1] - bb_df["BBL_5_2.0"].iloc[-1]
                prev_bb_width = bb_df["BBU_5_2.0"].iloc[-2] - bb_df["BBL_5_2.0"].iloc[-2] if len(bb_df) >= 2 else np.nan
                if pd.notna(prev_bb_width) and prev_bb_width > 0 and bb_width < prev_bb_width * 0.9: # BB 수렴 후 회복
                    surge_score += 0.5
                    if "BB 하단 & 중간선 돌파" not in temp_surge_signals:
                        temp_surge_signals.append("BB 수렴 후 회복")
                
                if not ("BB 하단 & 중간선 돌파" in temp_surge_signals or "BB 수렴 후 회복" in temp_surge_signals):
                    temp_surge_signals.append("BB 하단 회복")


        # 4. 거래량 점수 (최대 2.5점)
        is_volume_spike = False
        if pd.notna(avg_volume_20_days) and avg_volume_20_days > 0 and current_volume > (avg_volume_20_days * 2):
            is_volume_spike = True
            
            volume_signal_score = 0
            if current_volume > (avg_volume_20_days * 10):
                volume_signal_score += 1.5
                temp_surge_signals.append("거래량 10배+ 급증")
            elif current_volume > (avg_volume_20_days * 5):
                volume_signal_score += 1.0
                temp_surge_signals.append("거래량 5배+ 급증")
            elif current_volume > (avg_volume_20_days * 2):
                volume_signal_score += 0.5
                temp_surge_signals.append("거래량 2배+ 급증")
            
            surge_score += volume_signal_score # 기본 급증 점수

            if len(close_prices) >= 5: # 상승 추세에서 거래량 급증
                ma_5 = close_prices.rolling(window=5).mean().iloc[-1]
                prev_ma_5 = close_prices.rolling(window=5).mean().iloc[-2] if len(close_prices) >= 5 else np.nan
                if current_close_price > ma_5 and pd.notna(prev_ma_5) and ma_5 > prev_ma_5:
                    surge_score += 1.0 # 0.5점에서 1점으로 상향
                    temp_surge_signals.append("상승 추세 거래량")
        
        # 5. 외국인 수급 점수 (최대 1.5점)
        is_foreign_high_ownership = False
        if pd.notna(current_foreign_rate) and current_foreign_rate >= 5:
            is_foreign_high_ownership = True
            surge_score += 0.5 # 외국인 보유율 5% 이상 기본 점수
            
            if current_foreign_rate >= 10:
                surge_score += 0.5 # 외국인 보유율 10% 이상 추가 점수
                temp_surge_signals.append("외국인 보유율 10%+ 높은")
            
            if current_foreign_rate >= 20:
                surge_score += 0.5 # 외국인 보유율 20% 이상 추가 점수
                temp_surge_signals.append("외국인 보유율 20%+ 높은")
            
            if not ("외국인 보유율 10%+ 높은" in temp_surge_signals or "외국인 보유율 20%+ 높은" in temp_surge_signals):
                temp_surge_signals.append("외국인 보유율 5%+ 높은")


        # 급등 시그널 텍스트 정리 (최종적으로 유일한 시그널만 남기기)
        surge_signals_final = []
        
        # RSI 시그널 처리 (가장 강한 시그널 하나만)
        if "RSI 깊은 과매도 회복" in temp_surge_signals:
            surge_signals_final.append("RSI 깊은 과매도 회복")
        elif "RSI 50+ 돌파" in temp_surge_signals:
            surge_signals_final.append("RSI 50+ 돌파")
        elif "RSI 회복" in temp_surge_signals:
            surge_signals_final.append("RSI 회복")

        # MACD 시그널 처리 (가장 강한 시그널 하나만)
        if "MACD GC & 0선 돌파" in temp_surge_signals:
            surge_signals_final.append("MACD GC & 0선 돌파")
        elif "MACD 이격 확대" in temp_surge_signals:
            surge_signals_final.append("MACD 이격 확대")
        elif "MACD GC" in temp_surge_signals:
            surge_signals_final.append("MACD GC")

        # BB 시그널 처리 (가장 강한 시그널 하나만)
        if "BB 하단 & 중간선 돌파" in temp_surge_signals:
            surge_signals_final.append("BB 하단 & 중간선 돌파")
        elif "BB 수렴 후 회복" in temp_surge_signals:
            surge_signals_final.append("BB 수렴 후 회복")
        elif "BB 하단 회복" in temp_surge_signals:
            surge_signals_final.append("BB 하단 회복")

        # 거래량 시그널 처리 (가장 강한 시그널 하나 + 상승 추세)
        volume_signal_text = ""
        if "거래량 10배+ 급증" in temp_surge_signals:
            volume_signal_text = "거래량 10배+ 급증"
        elif "거래량 5배+ 급증" in temp_surge_signals:
            volume_signal_text = "거래량 5배+ 급증"
        elif "거래량 2배+ 급증" in temp_surge_signals:
            volume_signal_text = "거래량 2배+ 급증"
        
        if volume_signal_text:
            if "상승 추세 거래량" in temp_surge_signals:
                surge_signals_final.append(f"상승 추세 {volume_signal_text}")
            else:
                surge_signals_final.append(volume_signal_text)

        # 외국인 수급 시그널 처리 (가장 강한 시그널 하나만)
        if "외국인 보유율 20%+ 높은" in temp_surge_signals:
            surge_signals_final.append("외국인 보유율 20%+ 높은")
        elif "외국인 보유율 10%+ 높은" in temp_surge_signals:
            surge_signals_final.append("외국인 보유율 10%+ 높은")
        elif "외국인 보유율 5%+ 높은" in temp_surge_signals:
            surge_signals_final.append("외국인 보유율 5%+ 높은")

        surge_signal_text = ", ".join(surge_signals_final) if surge_signals_final else "없음"
        
        # --- ⭐ 급등_점수 10점 만점 세분화 로직 끝 ⭐ ---


        # ⭐ 총 급등 점수가 10점을 넘지 않도록 상한 적용 (선택 사항)
        # 현재 배점으로는 10점 초과가 가능할 수 있으므로, 최대 10점으로 제한하는 로직 추가
        surge_score = min(surge_score, 10.0)


        per_score = min(1/per * 100, 1)
        roe_score = roe / 20
        rsi_score_for_rank = 1 if current_rsi < 30 else 0 if current_rsi > 70 else 0.5 # 기존 랭크 점수용 RSI
        macd_score_for_rank = 1 if is_macd_golden_cross else 0 # 기존 랭크 점수용 MACD
        bb_score_for_rank = percent_b # 기존 랭크 점수용 BB
        foreign_score_for_rank = 1 if is_foreign_high_ownership else 0 # 기존 랭크 점수용 외국인 수급

        raw_metrics = [per_score, roe_score, rsi_score_for_rank, macd_score_for_rank, bb_score_for_rank, foreign_score_for_rank]

        naver_stock_link = f"https://finance.naver.com/item/main.naver?code={code}"

        estimated_return = np.nan # 기본값은 NaN
        target_per = 15.0 # ⭐ 예시: 목표 PER (산업 평균 PER 등으로 대체 가능)
        
        if pd.notna(per) and per > 0: # PER이 유효한 값일 때만 계산
            # PER이 너무 높거나 너무 낮은 극단적인 값일 경우 필터링 (예: PER 500 이상은 제외)
            if per < 500: # 500배 이상의 PER은 의미가 없을 수 있어서 필터링
                estimated_return = ((target_per / per) - 1) * 100
                estimated_return = round(estimated_return, 2)
            else:
                estimated_return = -100.0 # PER이 너무 높으면 큰 폭의 마이너스 수익률로 표시
        
        # PER이 0에 가깝거나 음수인 경우 (적자 기업 등) 처리
        if per <= 0:
            estimated_return = -999.99 # PER이 0이거나 음수인 경우 매우 낮은 수익률로 표시

        return {
            "종목명": name,
            "현재가": stock_info["현재가"],
            "PER": round(per, 2),
            "ROE": round(roe, 2),
            "RSI": round(current_rsi, 2) if pd.notna(current_rsi) else np.nan,
            "MACD_Line": round(macd_line, 2),
            "Signal_Line": round(signal_line, 2),
            "BB_Percent_B": round(percent_b, 2),
            "현재_거래량": int(current_volume) if pd.notna(current_volume) else 0,
            "20일_평균_거래량": int(avg_volume_20_days) if pd.notna(avg_volume_20_days) else 0,
            "외국인_현재_보유율": round(current_foreign_rate, 2) if pd.notna(current_foreign_rate) else np.nan,
            "외국인_보유율_변화": np.nan, 
            "급등_점수": round(surge_score, 2), # ⭐ 소수점 둘째자리까지 반올림
            "급등_시그널": surge_signal_text,
            "네이버_주식_링크": naver_stock_link,
            "예상_수익률": estimated_return,
            "_raw_metrics": raw_metrics
        }

    except Exception as e:
        print(f"⚠️ 종목 '{name}' (코드: {code}) 처리 중 예상치 못한 오류 발생: {e}")
        return None


# ✅ 종목 점수 계산 및 급등 필터 포함 (병렬 처리 적용)
def rank_stocks(df, code_map):
    all_stock_data = []
    
    max_workers = os.cpu_count() * 2 if os.cpu_count() else 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_stock, row.to_dict(), code_map): row['종목명'] for idx, row in df.iterrows()}
        
        processed_count = 0
        total_stocks = len(futures)

        for future in concurrent.futures.as_completed(futures):
            processed_count += 1
            stock_name = futures[future]
            try:
                result = future.result()
                if result:
                    all_stock_data.append(result)
            except Exception as exc:
                print(f"⚠️ 종목 '{stock_name}' 최종 결과 처리 중 예외 발생: {exc}")
            
            if processed_count % 100 == 0 or processed_count == total_stocks:
                print(f"  {processed_count}/{total_stocks} 종목 처리 완료 ({stock_name} 처리됨)")

    if not all_stock_data:
        print("❌ rank_stocks: 유효한 데이터를 가진 종목이 없어 최종 DataFrame이 비어있습니다. JSON 파일이 생성되지 않습니다.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        metrics_array = np.array([item["_raw_metrics"] for item in all_stock_data])
        scaler = MinMaxScaler()
        normed = scaler.fit_transform(metrics_array)
        
        weights = np.array([0.2, 0.2, 0.15, 0.15, 0.1, 0.2])
        final_scores = normed @ weights

        for i, score in enumerate(final_scores):
            all_stock_data[i]['score'] = round(score, 4)
            del all_stock_data[i]['_raw_metrics']
    except Exception as e:
        print(f"❌ 점수 정규화 및 계산 중 오류 발생: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


    full_df = pd.DataFrame(all_stock_data).sort_values(by="score", ascending=False).reset_index(drop=True)

    financial_df = full_df[
        (full_df['PER'] < 30) & (full_df['ROE'] > 5)
    ].sort_values(by=["ROE", "PER", "score"], ascending=[False, True, False]).head(50)

    # ⭐ 급등 점수 필터링 제거 및 정렬만 적용
    surge_df = full_df.sort_values(by="급등_점수", ascending=False).reset_index(drop=True)

    return full_df, financial_df, surge_df


# ✅ 시장별 실행
def run_for_market(market):
    print(f"\n--- 📊 {market.upper()} 시장 데이터 처리 시작 ---")
    start_time = time.time()
    
    # ⭐ 페이지 수를 5로 일단 줄여서 테스트해보자.
    df = fetch_stock_list(market, pages=5) 
    print(f"✅ {len(df)}개 종목 수집 완료.")
    if df.empty:
        print(f"⚠️ {market.upper()} 시장에 유효한 종목이 없어 분석을 건너뜁니다.")
        return

    print("🗺️ 종목 코드 매핑 중...")
    code_map = fetch_code_map()
    if not code_map:
        print(f"⚠️ 종목 코드 매핑에 실패했습니다. 분석을 건너뜁니다.")
        return

    print("⏳ 종목 점수 계산 중 (병렬 처리로 빠르게 진행됩니다)...")
    full, financial, surge = rank_stocks(df, code_map)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n✨ {market.upper()} 시장 데이터 처리 완료! (총 {elapsed_time:.2f}초 소요)")

    print(f"\n--- 🏆 {market.upper()} 전체 상위 종합 랭킹 ---")
    print(full[['종목명', '현재가', 'score', '예상_수익률', '네이버_주식_링크']].head(20) if not full.empty else "없음")

    print(f"\n--- 💰 {market.upper()} 재무 건전성 우수 종목 (PER 30 이하, ROE 5 이상 기준) ---")
    print(financial[['종목명', '현재가', 'PER', 'ROE', '예상_수익률', 'score', '네이버_주식_링크']].head(20) if not financial.empty else "없음")

    print(f"\n--- 🔥 {market.upper()} 급등 가능성 종목 (급등 점수 순) ---") # 출력 문구 변경
    # ⭐ 급등 점수 상위 20개 종목 출력
    print(surge[['종목명', '현재가', '급등_점수', '급등_시그널', '현재_거래량', '20일_평균_거래량', '외국인_현재_보유율', '예상_수익률', 'score', '네이버_주식_링크']].head(20) if not surge.empty else "없음") 

    # 저장 디렉토리 생성
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if not full.empty:
        full.to_json(f"{output_dir}/{market}_ranked_stocks_all.json", orient="records", force_ascii=False, indent=2)
        print(f"\n📁 {output_dir}/{market}_ranked_stocks_all.json 저장 완료")
    else:
        print(f"⚠️ {market.upper()} 전체 랭킹 데이터가 비어있어 JSON 파일 저장 건너뜁니다.")
    
    if not financial.empty:
        financial.to_json(f"{output_dir}/{market}_ranked_stocks_financial_healthy.json", orient="records", force_ascii=False, indent=2)
        print(f"📁 {output_dir}/{market}_ranked_stocks_financial_healthy.json 저장 완료")
    else:
        print(f"⚠️ {market.upper()} 재무 건전성 데이터가 비어있어 JSON 파일 저장 건너킵니다.")

    if not surge.empty:
        # ⭐ 급등 점수 필터링을 없앴으니, 파일명도 필터 기준이 없다는 것을 반영 (potential_surge 유지 가능)
        surge.to_json(f"{output_dir}/{market}_ranked_stocks_potential_surge.json", orient="records", force_ascii=False, indent=2)
        print(f"📁 {output_dir}/{market}_ranked_stocks_potential_surge.json 저장 완료")
    else:
        print(f"⚠️ {market.upper()} 급등 가능성 데이터가 비어있어 JSON 파일 저장 건너킵니다.")
    
    print(f"\n--- {market.upper()} 시장 데이터 처리 완료 ---\n")


# ✅ 실행
if __name__ == "__main__":
    run_for_market("kospi")
    run_for_market("kosdaq")