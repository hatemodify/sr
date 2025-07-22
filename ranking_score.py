import os
import platform
import subprocess
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from io import StringIO
import concurrent.futures
import time

# --- 유틸리티 함수 ---
def safe_float(value):
    """콤마가 포함된 문자열 등을 안전하게 float으로 변환. 실패 시 np.nan 반환"""
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, AttributeError):
        return np.nan

# --- 데이터 수집 함수 ---

def fetch_code_map():
    """KRX에서 전체 종목 코드를 다운로드하여 '회사명: 종목코드' 딕셔너리로 반환"""
    url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
    try:
        # read_html은 user-agent를 직접 설정하기 어려우므로, requests로 먼저 데이터를 가져옴
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=20)
        res.raise_for_status() # HTTP 오류 발생 시 예외 발생
        
        # euc-kr로 디코딩
        html_content = res.content.decode('euc-kr')
        
        df = pd.read_html(StringIO(html_content))[0]
        df = df[['회사명', '종목코드']]
        # 종목코드를 6자리 문자열로 포맷팅 (예: 5930 -> '005930')
        # 숫자로 변환 가능한 경우에만 포맷팅 적용
        df['종목코드'] = df['종목코드'].apply(lambda x: f"{int(x):06d}" if pd.notna(x) and str(x).isdigit() else str(x))
        return dict(zip(df['회사명'], df['종목코드']))
    except Exception as e:
        print(f"❌ KRX 종목 코드 매핑 실패: {e}")
        return {}

def fetch_real_close_prices(code, days=120):
    """네이버 금융에서 일별 시세(종가, 거래량)를 스크래핑"""
    sise_url = f"https://finance.naver.com/item/sise_day.nhn?code={code}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    
    all_rows = []
    # 최대 10페이지 (약 100거래일)까지만 시도하여 너무 많은 요청 방지
    for page in range(1, 11): 
        pg_url = f"{sise_url}&page={page}"
        try:
            res = requests.get(pg_url, headers=headers, timeout=5)
            if res.status_code != 200:
                break
            
            if "캡차" in res.text or "차단" in res.text:
                print(f"⚠️ {code} 시세 크롤링 중 캡차/IP 차단 감지 (페이지: {page}).")
                break

            # pandas.read_html은 페이지의 모든 테이블을 리스트로 반환
            tables = pd.read_html(StringIO(res.text), header=0) # header=0으로 첫 행을 컬럼으로 지정
            if not tables:
                break

            # '날짜' 컬럼이 있는 테이블이 우리가 원하는 시세 테이블
            df_page = None
            for table in tables:
                if '날짜' in table.columns:
                    df_page = table
                    break
            
            if df_page is None or df_page.empty:
                break

            all_rows.append(df_page.dropna())
            time.sleep(0.1) # 요청 간 최소한의 딜레이
        except (requests.exceptions.RequestException, ValueError, IndexError) as e:
            break

    if not all_rows:
        return pd.Series([], dtype=float), pd.Series([], dtype=float)

    df_sise = pd.concat(all_rows, ignore_index=True)
    df_sise = df_sise.dropna(subset=['날짜']) # 날짜가 없는 행 제거
    df_sise = df_sise.drop_duplicates().sort_values(by='날짜')
    
    close_prices = df_sise['종가'].astype(float).tail(days)
    volumes = df_sise['거래량'].astype(float).tail(days)
    
    return close_prices, volumes


def fetch_stock_list(market='kospi', pages=10):
    """네이버 금융에서 시장별(코스피/코스닥) 종목 목록 및 기본 정보를 스크래핑"""
    base_url = {
        'kospi': 'https://finance.naver.com/sise/sise_market_sum.nhn?sosok=0&page=',
        'kosdaq': 'https://finance.naver.com/sise/sise_market_sum.nhn?sosok=1&page='
    }[market]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    
    all_stocks = []
    for page in range(1, pages + 1):
        url = base_url + str(page)
        print(f"--- 🌐 {market.upper()} 페이지 {page} 크롤링 시도 중 ---")
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code != 200:
                print(f"⚠️ 페이지 {page} 접속 실패: Status Code {res.status_code}")
                continue
            
            if "캡차" in res.text or "차단" in res.text or "bot_block" in res.text:
                print(f"❌ {market.upper()} 시장 크롤링 중 캡차/IP 차단 감지 (페이지: {page}). 중단합니다.")
                break

            tables = pd.read_html(StringIO(res.text))
            df_page = None
            # '종목명' 컬럼이 있는 테이블을 찾음 (보통 시가총액 테이블)
            for table in tables:
                if '종목명' in table.columns:
                    df_page = table
                    break
            
            if df_page is None:
                print(f"⚠️ 페이지 {page}에서 유효한 종목 테이블을 찾지 못했습니다.")
                continue

            # 불필요한 행과 컬럼 제거
            df_page = df_page.dropna(subset=['종목명'])
            # 'N' 컬럼은 순번이므로 제거, '토론실' 링크도 불필요
            df_page = df_page.drop(columns=[col for col in ['N', '토론실'] if col in df_page.columns])

            # 컬럼명 통일 (외국인비율 -> 외국인보유율)
            df_page = df_page.rename(columns={'외국인비율': '외국인보유율'})
            
            # 필요한 컬럼만 선택
            required_cols = ['종목명', '현재가', 'PER', 'ROE', '외국인보유율']
            if not all(col in df_page.columns for col in required_cols):
                print(f"⚠️ 페이지 {page}에 필수 컬럼 {required_cols}이(가) 없습니다. 건너뜁니다.")
                continue
            
            df_page = df_page[required_cols].copy()

            # 데이터 타입 변환 및 정리
            for col in ['현재가', 'PER', 'ROE', '외국인보유율']:
                df_page[col] = df_page[col].apply(safe_float)
            
            df_page = df_page.dropna()
            
            if df_page.empty:
                print(f"⚠️ 페이지 {page} 정리 후 데이터가 없습니다.")
                continue

            all_stocks.append(df_page)
            print(f"✅ 페이지 {page}에서 {len(df_page)}개 종목 수집 완료.")
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ 시장 목록 크롤링 오류 (페이지 {page}): {e}")
            continue

    if not all_stocks:
        return pd.DataFrame()

    full_df = pd.concat(all_stocks, ignore_index=True)
    return full_df.drop_duplicates(subset=['종목명']).reset_index(drop=True)


# --- 데이터 분석 및 처리 함수 ---

def _process_single_stock(stock_info, code_map):
    """개별 종목 정보를 받아 기술적 지표 및 점수를 계산"""
    name = stock_info['종목명']
    code = code_map.get(name)
    if not code:
        return None

    try:
        close_prices, volumes = fetch_real_close_prices(code)
        
        if len(close_prices) < 60 or len(volumes) < 60:
            return None

        # 기술적 지표 계산
        rsi = ta.rsi(close_prices, length=14).iloc[-1]
        macd_df = ta.macd(close_prices, fast=12, slow=26, signal=9)
        macd_line, signal_line = macd_df.iloc[-1, 0], macd_df.iloc[-1, 1]
        bb_df = ta.bbands(close_prices, length=20, std=2) # BBands는 보통 20일 기준
        percent_b = bb_df[f'BBP_{20}_{2.0}'].iloc[-1]

        # 이전 값들
        prev_rsi = ta.rsi(close_prices, length=14).iloc[-2]
        prev_macd_line, prev_signal_line = macd_df.iloc[-2, 0], macd_df.iloc[-2, 1]

        # 거래량
        current_volume = volumes.iloc[-1]
        avg_volume_20 = volumes.rolling(window=20).mean().iloc[-1]

        # --- 급등 점수 계산 로직 (형의 로직 유지) ---
        surge_score = 0
        surge_signals = []

        # 1. RSI (과매도 탈출)
        if prev_rsi < 30 and rsi >= 30:
            surge_score += 2.5
            surge_signals.append("RSI 과매도 탈출")

        # 2. MACD (골든 크로스)
        if prev_macd_line <= prev_signal_line and macd_line > signal_line:
            surge_score += 2.5
            surge_signals.append("MACD 골든크로스")

        # 3. Bollinger Bands (%B < 0.2 에서 회복)
        if percent_b < 0.2: # %B가 0.2 미만이면 과매도 구간으로 간주
            # 여기서는 실제 회복 시그널을 잡으려면 이전 값을 비교해야 하지만, 단순화하여 낮은 상태 자체에 점수 부여
            surge_score += 1.5
            surge_signals.append("BB 하단 근접")

        # 4. 거래량 급증 (20일 평균 대비)
        if pd.notna(current_volume) and pd.notna(avg_volume_20) and avg_volume_20 > 0:
            if current_volume > avg_volume_20 * 5:
                surge_score += 2.5
                surge_signals.append("거래량 5배↑ 급증")
            elif current_volume > avg_volume_20 * 2:
                surge_score += 1.0
                surge_signals.append("거래량 2배↑")
        
        # 5. 외국인 보유율
        current_foreign_rate = stock_info.get('외국인보유율', 0)
        if current_foreign_rate > 5:
            surge_score += 1.0
            surge_signals.append("외국인 보유율 5% 이상")
        
        surge_score = min(surge_score, 10.0) # 최대 10점으로 제한

        # --- 종합 순위 점수 계산용 지표 ---
        per = stock_info["PER"]
        roe = stock_info["ROE"]
        per_score = 1 / per if per > 0 else 0
        roe_score = roe / 100 if roe > 0 else 0
        
        # 순위 점수용 지표는 정규화를 위해 raw 값으로 저장
        raw_metrics = [
            per_score,
            roe_score,
            1 if rsi < 30 else 0, # 과매도
            1 if macd_line > signal_line else 0, # 상승추세
            percent_b,
            current_foreign_rate / 100
        ]

        # 예상 수익률 계산
        estimated_return = np.nan
        target_per = 15.0 # 목표 PER
        if per > 0 and per < 500:
            estimated_return = round(((target_per / per) - 1) * 100, 2)
        elif per <= 0:
             estimated_return = -999.99 # 적자 기업

        return {
            "종목명": name,
            "현재가": int(stock_info["현재가"]),
            "PER": round(per, 2),
            "ROE": round(roe, 2),
            "RSI": round(rsi, 2),
            "BB_Percent_B": round(percent_b, 2),
            "현재_거래량": int(current_volume),
            "20일_평균_거래량": int(avg_volume_20),
            "외국인_현재_보유율": round(current_foreign_rate, 2),
            "급등_점수": round(surge_score, 2),
            "급등_시그널": ", ".join(surge_signals) if surge_signals else "없음",
            "예상_수익률": estimated_return,
            "네이버_주식_링크": f"https://finance.naver.com/item/main.naver?code={code}",
            "_raw_metrics": raw_metrics
        }

    except Exception as e:
        # print(f"⚠️ 종목 '{name}' ({code}) 처리 중 오류: {e}") # 디버깅 시 주석 해제
        return None


def rank_stocks(df, code_map):
    """수집된 종목 리스트를 병렬 처리하여 점수 계산 및 랭킹 산정"""
    all_stock_data = []
    
    max_workers = min(32, (os.cpu_count() or 1) + 4) # 너무 많은 스레드 방지

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_stock, row.to_dict(), code_map): row['종목명'] for _, row in df.iterrows()}
        
        total = len(futures)
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            stock_name = futures[future]
            try:
                result = future.result()
                if result:
                    all_stock_data.append(result)
            except Exception as exc:
                print(f"⚠️ '{stock_name}' 처리 중 예외 발생: {exc}")
            
            # 진행 상황 표시
            print(f"\r⏳ 종목 점수 계산 진행: {i}/{total} ({i/total:.1%})", end="")

    print("\n✅ 종목 점수 계산 완료.")

    if not all_stock_data:
        print("❌ 유효한 분석 데이터가 없습니다.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 종합 점수 계산 (Min-Max Scaling)
    try:
        metrics_array = np.array([item["_raw_metrics"] for item in all_stock_data])
        scaler = MinMaxScaler()
        normed_metrics = scaler.fit_transform(metrics_array)
        
        # 가중치: PER, ROE, RSI, MACD, BB, 외국인
        weights = np.array([0.25, 0.25, 0.15, 0.15, 0.1, 0.1])
        final_scores = normed_metrics @ weights

        for i, score in enumerate(final_scores):
            all_stock_data[i]['종합점수'] = round(score * 100, 2)
            del all_stock_data[i]['_raw_metrics']
            
    except Exception as e:
        print(f"❌ 점수 정규화 및 계산 중 오류: {e}")
        # 오류 발생 시 종합점수 없이 진행
        for item in all_stock_data:
            item['종합점수'] = np.nan
            if '_raw_metrics' in item:
                del item['_raw_metrics']


    full_df = pd.DataFrame(all_stock_data).sort_values(by="종합점수", ascending=False).reset_index(drop=True)

    # 1. 재무 건전성 필터링
    financial_df = full_df[
        (full_df['PER'] > 0) & (full_df['PER'] < 20) & (full_df['ROE'] > 8)
    ].sort_values(by=["ROE", "PER"], ascending=[False, True]).head(50)

    # 2. 급등 가능성 필터링 (급등 점수 높은 순)
    surge_df = full_df.sort_values(by="급등_점수", ascending=False).reset_index(drop=True)

    return full_df, financial_df, surge_df


# --- 메인 실행 함수 ---

def run_for_market(market):
    """시장별로 전체 분석 파이프라인 실행"""
    print(f"\n{'='*50}\n--- 📊 {market.upper()} 시장 데이터 분석 시작 ---\n{'='*50}")
    start_time = time.time()
    
    df = fetch_stock_list(market, pages=5) # 테스트를 위해 페이지 수 5로 제한
    if df.empty:
        print(f"⚠️ {market.upper()} 시장 종목 수집 실패. 분석을 건너뜁니다.")
        return

    print(f"\n🗺️ KRX 전체 종목 코드 매핑 중...")
    code_map = fetch_code_map()
    if not code_map:
        print(f"⚠️ 종목 코드 매핑 실패. 분석을 건너뜁니다.")
        return
    print("✅ 종목 코드 매핑 완료.")

    full, financial, surge = rank_stocks(df, code_map)

    end_time = time.time()
    print(f"\n✨ {market.upper()} 시장 분석 완료! (총 {end_time - start_time:.2f}초 소요)")

    # 결과 출력
    output_cols = ['종목명', '현재가', '종합점수', '예상_수익률', '네이버_주식_링크']
    print(f"\n--- 🏆 {market.upper()} 전체 상위 20 ---")
    print(full[output_cols].head(20).to_string() if not full.empty else "데이터 없음")

    financial_cols = ['종목명', '현재가', 'PER', 'ROE', '종합점수', '예상_수익률', '네이버_주식_링크']
    print(f"\n--- 💰 {market.upper()} 재무 우수 상위 20 ---")
    print(financial[financial_cols].head(20).to_string() if not financial.empty else "데이터 없음")

    surge_cols = ['종목명', '현재가', '급등_점수', '급등_시그널', '종합점수', '예상_수익률', '네이버_주식_링크']
    print(f"\n--- 🔥 {market.upper()} 급등 시그널 상위 20 ---")
    print(surge[surge_cols].head(20).to_string() if not surge.empty else "데이터 없음")

    # 파일 저장
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    if not full.empty:
        full.to_json(f"{output_dir}/{market}_ranked_all.json", orient="records", force_ascii=False, indent=2)
        print(f"\n📁 {output_dir}/{market}_ranked_all.json 저장 완료")
    if not financial.empty:
        financial.to_json(f"{output_dir}/{market}_ranked_financial.json", orient="records", force_ascii=False, indent=2)
        print(f"📁 {output_dir}/{market}_ranked_financial.json 저장 완료")
    if not surge.empty:
        surge.to_json(f"{output_dir}/{market}_ranked_surge.json", orient="records", force_ascii=False, indent=2)
        print(f"📁 {output_dir}/{market}_ranked_surge.json 저장 완료")


if __name__ == "__main__":
    run_for_market("kospi")
    run_for_market("kosdaq")