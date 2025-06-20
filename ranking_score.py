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

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    
    close_prices = pd.Series([], dtype=float)
    volumes = pd.Series([], dtype=float)

    # 종가 및 거래량 데이터 크롤링
    dfs_sise = []
    for page in range(1, 10):
        pg_url = f"{sise_url}&page={page}"
        try:
            res = requests.get(pg_url, headers=headers, timeout=10)
            if res.status_code != 200:
                break
            df = pd.read_html(StringIO(res.text))[0]
            if df.empty:
                break
            dfs_sise.append(df)
        except (requests.exceptions.RequestException, ValueError):
            break

    if dfs_sise:
        df_sise = pd.concat(dfs_sise, ignore_index=True)
        df_sise = df_sise.dropna()
        df_sise['날짜'] = pd.to_datetime(df_sise['날짜'])
        df_sise = df_sise.sort_values(by='날짜').reset_index(drop=True)
        close_prices = df_sise['종가'].astype(float).tail(days)
        volumes = df_sise['거래량'].astype(float).tail(days)
    
    return close_prices, volumes


# ✅ 시장별 주식 목록 및 기본 정보 수집 (외국인비율 컬럼 추가)
def fetch_stock_list(market='kospi', pages=10):
    base_url = {
        'kospi': 'https://finance.naver.com/sise/sise_market_sum.nhn?sosok=0&page=',
        'kosdaq': 'https://finance.naver.com/sise/sise_market_sum.nhn?sosok=1&page='
    }[market]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    dfs = []
    required_cols = ['종목명', '현재가', 'PER', 'ROE'] # 필수 컬럼
    foreign_rate_col_candidates = ['외국인비율', '외국인비율(%)', '외국인보유율'] # 외국인 비율 컬럼 후보들

    for page in range(1, pages + 1):
        url = base_url + str(page)
        print(f"--- 🌐 {market.upper()} 페이지 {page} 크롤링 시도 중 ---") # 디버깅용 print
        try:
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code != 200:
                print(f"⚠️ 페이지 {page} 접속 실패: Status Code {res.status_code}")
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            table_selector = "table.type_2"
            table = soup.select_one(table_selector)
            if not table:
                print(f"⚠️ 페이지 {page}에서 '{table_selector}' 테이블을 찾을 수 없습니다.")
                continue
            
            tables_from_html = pd.read_html(StringIO(res.text))
            df_page = pd.DataFrame()
            found_target_table = False

            for temp_df in tables_from_html:
                has_all_required_base_cols = all(col in temp_df.columns for col in required_cols)
                has_foreign_rate_col = any(col in temp_df.columns for col in foreign_rate_col_candidates)
                
                if has_all_required_base_cols and has_foreign_rate_col:
                    df_page = temp_df
                    found_target_table = True
                    break
            
            if not found_target_table:
                print(f"⚠️ 페이지 {page}에서 필수 컬럼을 포함하는 테이블을 찾을 수 없습니다.")
                continue 

            foreign_rate_col_name_found = None
            for col_candidate in foreign_rate_col_candidates:
                if col_candidate in df_page.columns:
                    foreign_rate_col_name_found = col_candidate
                    break
            
            if not foreign_rate_col_name_found:
                print(f"⚠️ 페이지 {page}에서 외국인비율 컬럼을 찾을 수 없습니다. (후보: {foreign_rate_col_candidates}, 실제: {df_page.columns.tolist()})")
                continue 

            df_page = df_page.rename(columns={foreign_rate_col_name_found: '외국인비율'})
            
            final_cols = required_cols + ['외국인비율']

            if not all(col in df_page.columns for col in final_cols):
                missing_cols = [col for col in final_cols if col not in df_page.columns]
                print(f"⚠️ 페이지 {page} - 최종 컬럼 선택 후 누락된 컬럼: {missing_cols}. 건너뜁니다.")
                continue

            df_page = df_page[final_cols].copy() 

            # ⭐ 수정된 부분: safe_float를 먼저 적용하여 float으로 변환
            df_page['현재가'] = df_page['현재가'].apply(safe_float)
            df_page['PER'] = df_page['PER'].apply(safe_float)
            df_page['ROE'] = df_page['ROE'].apply(safe_float)
            df_page['외국인비율'] = df_page['외국인비율'].apply(safe_float)

            # NaN 값 제거 (여기서 제거하면 KeyError가 나지 않도록 이미 컬럼 존재 여부 확인 후)
            # 숫자 변환 오류가 아니라 데이터 자체에 문제가 있는 경우를 위해 다시 한번 dropna
            df_page = df_page.dropna(subset=final_cols)
            
            # ⭐ 수정된 부분: 현재가는 정수형이므로 다시 int로 변환 (NaN 값 제거 후)
            df_page['현재가'] = df_page['현재가'].astype(int)

            if df_page.empty:
                print(f"⚠️ 페이지 {page} - NaN 값 제거 후 데이터가 없습니다. 건너뜁니다.")
                continue
            
            dfs.append(df_page)
            print(f"✅ 페이지 {page}에서 {len(df_page)}개 종목 수집 완료.") # 성공 시 출력

        except (requests.exceptions.RequestException, ValueError, IndexError, KeyError, TypeError) as e:
            print(f"❌ 시장 목록 크롤링 오류 (페이지 {page}, 상세: {e})")
            continue 

    if not dfs:
        print("❌ 시장 목록을 가져오는 데 실패했습니다. 빈 DataFrame 반환.")
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna(subset=required_cols + ['외국인비율']) 
    
    return full_df.reset_index(drop=True)


# 각 종목을 처리하는 헬퍼 함수 (변경 없음)
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
        current_close_price = close_prices.iloc[-1]
        current_volume = volumes.iloc[-1]
        
        avg_volume_20_days = volumes.rolling(window=20).mean().iloc[-1] if len(volumes) >= 20 else np.nan

        foreign_ownership_change = np.nan 
        
        is_foreign_inflow = True if pd.notna(current_foreign_rate) and current_foreign_rate >= 5 else False 

        per = stock_info["PER"] if pd.notna(stock_info["PER"]) and stock_info["PER"] > 0 else 1000
        roe = stock_info["ROE"] if pd.notna(stock_info["ROE"]) else 0

        is_rsi_recovery = False
        if pd.notna(prev_rsi) and prev_rsi < 30 and current_rsi >= 30:
            is_rsi_recovery = True

        is_macd_golden_cross = False
        if pd.notna(prev_macd_line) and pd.notna(prev_signal_line) and \
           macd_line > signal_line and prev_macd_line <= prev_signal_line:
            is_macd_golden_cross = True
        
        is_bb_lower_band_recovery = False
        if len(close_prices) >= 10 and not bb_df["BBL_5_2.0"].iloc[-10:].isnull().any():
            was_below_bbl = any(close_prices.iloc[-10:-1] < bb_df["BBL_5_2.0"].iloc[-10:-1])
            is_above_bbl_now = current_close_price > bbl
            if was_below_bbl and is_above_bbl_now:
                is_bb_lower_band_recovery = True

        is_volume_spike = False
        if pd.notna(avg_volume_20_days) and current_volume > (avg_volume_20_days * 2):
            is_volume_spike = True
        
        is_foreign_inflow = True if pd.notna(current_foreign_rate) and current_foreign_rate >= 5 else False 

        surge_score = sum([
            is_rsi_recovery,
            is_macd_golden_cross,
            is_bb_lower_band_recovery,
            is_volume_spike,
            is_foreign_inflow 
        ])
        surge_signal_text = ", ".join([
            s for s, flag in {
                "RSI 회복": is_rsi_recovery,
                "MACD GC": is_macd_golden_cross,
                "BB 회복": is_bb_lower_band_recovery,
                "거래량 급증": is_volume_spike,
                "외국인 수급(비율 높음)": is_foreign_inflow 
            }.items() if flag
        ])
        if not surge_signal_text:
            surge_signal_text = "없음"

        per_score = min(1/per * 100, 1)
        roe_score = roe / 20
        rsi_score = 1 if current_rsi < 30 else 0 if current_rsi > 70 else 0.5
        macd_score = 1 if is_macd_golden_cross else 0
        bb_score = percent_b
        foreign_score = 1 if is_foreign_inflow else 0 

        raw_metrics = [per_score, roe_score, rsi_score, macd_score, bb_score, foreign_score]

        naver_stock_link = f"https://finance.naver.com/item/main.naver?code={code}"

        return {
            "종목명": name,
            "현재가": stock_info["현재가"],
            "PER": round(per, 2),
            "ROE": round(roe, 2),
            "RSI": round(current_rsi, 2),
            "MACD_Line": round(macd_line, 2),
            "Signal_Line": round(signal_line, 2),
            "BB_Percent_B": round(percent_b, 2),
            "현재_거래량": int(current_volume),
            "20일_평균_거래량": int(avg_volume_20_days) if pd.notna(avg_volume_20_days) else 0,
            "외국인_현재_보유율": round(current_foreign_rate, 2) if pd.notna(current_foreign_rate) else np.nan,
            "외국인_보유율_변화": np.nan, 
            "급등_점수": surge_score,
            "급등_시그널": surge_signal_text,
            "네이버_주식_링크": naver_stock_link,
            "_raw_metrics": raw_metrics
        }

    except Exception as e:
        print(f"⚠️ 종목 '{name}' (코드: {code}) 처리 중 예상치 못한 오류 발생: {e}")
        return None


# ✅ 종목 점수 계산 및 급등 필터 포함 (병렬 처리 적용) (변경 없음)
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

    surge_df = full_df[full_df['급등_점수'] >= 2].sort_values(by="급등_점수", ascending=False).reset_index(drop=True)

    return full_df, financial_df, surge_df


# ✅ 시장별 실행 (변경 없음)
def run_for_market(market):
    print(f"\n--- 📊 {market.upper()} 시장 데이터 처리 시작 ---")
    start_time = time.time()
    
    df = fetch_stock_list(market, pages=20) 
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
    print(full[['종목명', '현재가', 'score', '네이버_주식_링크']].head(20) if not full.empty else "없음")

    print(f"\n--- 💰 {market.upper()} 재무 건전성 우수 종목 (PER 30 이하, ROE 5 이상 기준) ---")
    print(financial[['종목명', '현재가', 'PER', 'ROE', 'score', '네이버_주식_링크']].head(20) if not financial.empty else "없음")

    print(f"\n--- 🔥 {market.upper()} 급등 가능성 종목 (급등_점수 ≥ 2) ---")
    print(surge[['종목명', '현재가', '급등_점수', '급등_시그널', '현재_거래량', '20일_평균_거래량', '외국인_현재_보유율', 'score', '네이버_주식_링크']].head(20) if not surge.empty else "없음") 

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
        print(f"⚠️ {market.upper()} 재무 건전성 데이터가 비어있어 JSON 파일 저장 건너뜁니다.")

    if not surge.empty:
        surge.to_json(f"{output_dir}/{market}_ranked_stocks_potential_surge.json", orient="records", force_ascii=False, indent=2)
        print(f"📁 {output_dir}/{market}_ranked_stocks_potential_surge.json 저장 완료")
    else:
        print(f"⚠️ {market.upper()} 급등 가능성 데이터가 비어있어 JSON 파일 저장 건너뜁니다.")
    
    print(f"\n--- {market.upper()} 시장 데이터 처리 완료 ---\n")


# ✅ 실행
if __name__ == "__main__":
    run_for_market("kospi")
    run_for_market("kosdaq")