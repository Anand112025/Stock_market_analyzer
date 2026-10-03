import warnings
import logging
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

def safe_number(value):
    try:
        if value is None:
            return np.nan
        value = float(value)
        if np.isfinite(value):
            return value
        return np.nan
    except:
        return np.nan

def safe_pct_change(current, previous):
    current = safe_number(current)
    previous = safe_number(previous)
    if pd.isna(current) or pd.isna(previous):
        return np.nan
    if previous == 0:
        return np.nan
    return (current - previous) / abs(previous) * 100

def safe_divide(a, b):
    a = safe_number(a)
    b = safe_number(b)
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b

def safe_row(statement, possible_names):
    if statement is None or statement.empty:
        return None
    for name in possible_names:
        if name in statement.index:
            return statement.loc[name]
    return None

def safe_info_value(info, key):
    try:
        value = info.get(key, np.nan)
        return safe_number(value)
    except:
        return np.nan

def calculate_trend(values):
    clean_values = []
    for value in values:
        value = safe_number(value)
        if pd.notna(value):
            clean_values.append(value)
    if len(clean_values) < 2:
        return 'NO_DATA'
    increases = 0
    decreases = 0
    for i in range(1, len(clean_values)):
        if clean_values[i] > clean_values[i - 1]:
            increases += 1
        elif clean_values[i] < clean_values[i - 1]:
            decreases += 1
    if increases > decreases:
        return 'INCREASING'
    elif decreases > increases:
        return 'DECREASING'
    else:
        return 'MIXED'

def get_sorted_columns(statement):
    if statement is None or statement.empty:
        return []
    columns = []
    for column in statement.columns:
        try:
            date = pd.to_datetime(column)
            if pd.notna(date):
                columns.append((date, column))
        except:
            continue
    columns = sorted(columns, key=lambda x: x[0], reverse=True)
    return columns

def analyze_company_step32(company, symbol):
    result = {'Company': company, 'Symbol': symbol, 'Previous Close': np.nan, 'Current Price': np.nan, '1D Price Change %': np.nan, 'Price Status': 'NOT_AVAILABLE', 'Market Cap': np.nan, 'Trailing P/E': np.nan, 'Forward P/E': np.nan, 'Price / Book': np.nan, 'Enterprise Value': np.nan, 'EV / EBITDA': np.nan, 'Dividend Yield %': np.nan, '52W High': np.nan, '52W Low': np.nan, 'Distance From 52W High %': np.nan, 'Distance From 52W Low %': np.nan, 'Valuation Status': 'NOT_AVAILABLE', 'Latest Quarter Date': None, 'Previous Quarter Date': None, 'Quarter -3 Date': None, 'Quarter -4 Date': None, 'Revenue Q1': np.nan, 'Revenue Q2': np.nan, 'Revenue Q3': np.nan, 'Revenue Q4': np.nan, 'Revenue Q2 vs Q1 %': np.nan, 'Revenue Q3 vs Q2 %': np.nan, 'Revenue Q4 vs Q3 %': np.nan, 'Revenue Q4 YoY %': np.nan, 'Revenue Trend': 'NO_DATA', 'Profit Q1': np.nan, 'Profit Q2': np.nan, 'Profit Q3': np.nan, 'Profit Q4': np.nan, 'Profit Q2 vs Q1 %': np.nan, 'Profit Q3 vs Q2 %': np.nan, 'Profit Q4 vs Q3 %': np.nan, 'Profit Q4 YoY %': np.nan, 'Profit Trend': 'NO_DATA', 'EBITDA Q1': np.nan, 'EBITDA Q2': np.nan, 'EBITDA Q3': np.nan, 'EBITDA Q4': np.nan, 'EBITDA Q2 vs Q1 %': np.nan, 'EBITDA Q3 vs Q2 %': np.nan, 'EBITDA Q4 vs Q3 %': np.nan, 'EBITDA Q4 YoY %': np.nan, 'EBITDA Trend': 'NO_DATA', 'Net Margin Q1 %': np.nan, 'Net Margin Q2 %': np.nan, 'Net Margin Q3 %': np.nan, 'Net Margin Q4 %': np.nan, 'Net Margin Q4 vs Q3 Change': np.nan, 'EBITDA Margin Q1 %': np.nan, 'EBITDA Margin Q2 %': np.nan, 'EBITDA Margin Q3 %': np.nan, 'EBITDA Margin Q4 %': np.nan, 'EBITDA Margin Q4 vs Q3 Change': np.nan, 'Current Debt': np.nan, 'Previous Quarter Debt': np.nan, 'Debt Q1': np.nan, 'Debt Q2': np.nan, 'Debt Q3': np.nan, 'Debt Q4': np.nan, 'Debt Q2 vs Q1 %': np.nan, 'Debt Q3 vs Q2 %': np.nan, 'Debt Q4 vs Q3 %': np.nan, 'Debt Change %': np.nan, 'Debt Trend': 'NO_DATA', 'Debt Status': 'NOT_AVAILABLE', 'Cash Q1': np.nan, 'Cash Q2': np.nan, 'Cash Q3': np.nan, 'Cash Q4': np.nan, 'Cash Q4 vs Q3 %': np.nan, 'Cash Trend': 'NO_DATA', 'Net Debt Q1': np.nan, 'Net Debt Q2': np.nan, 'Net Debt Q3': np.nan, 'Net Debt Q4': np.nan, 'Net Debt Q4 vs Q3 %': np.nan, 'Net Debt Trend': 'NO_DATA', 'Operating CF Q1': np.nan, 'Operating CF Q2': np.nan, 'Operating CF Q3': np.nan, 'Operating CF Q4': np.nan, 'Operating CF Q4 vs Q3 %': np.nan, 'Operating CF Trend': 'NO_DATA', 'FCF Q1': np.nan, 'FCF Q2': np.nan, 'FCF Q3': np.nan, 'FCF Q4': np.nan, 'FCF Q4 vs Q3 %': np.nan, 'FCF Trend': 'NO_DATA', 'Next Result Date': None, 'Result Date Status': 'NOT_AVAILABLE', 'Quarterly Status': 'NO_DATA', 'Overall Data Status': 'NO_DATA'}
    try:
        ticker = yf.Ticker(symbol)
    except:
        return result
    try:
        history = ticker.history(period='5d', auto_adjust=False)
        if history is not None and (not history.empty):
            close = pd.to_numeric(history['Close'], errors='coerce').dropna()
            if len(close) >= 1:
                result['Current Price'] = close.iloc[-1]
                result['Price Status'] = 'AVAILABLE'
            if len(close) >= 2:
                result['Previous Close'] = close.iloc[-2]
                result['1D Price Change %'] = safe_pct_change(close.iloc[-1], close.iloc[-2])
    except:
        pass
    try:
        info = ticker.info
        result['Market Cap'] = safe_info_value(info, 'marketCap')
        result['Trailing P/E'] = safe_info_value(info, 'trailingPE')
        result['Forward P/E'] = safe_info_value(info, 'forwardPE')
        result['Price / Book'] = safe_info_value(info, 'priceToBook')
        result['Enterprise Value'] = safe_info_value(info, 'enterpriseValue')
        result['EV / EBITDA'] = safe_info_value(info, 'enterpriseToEbitda')
        dividend_yield = safe_info_value(info, 'dividendYield')
        if pd.notna(dividend_yield):
            if dividend_yield < 1:
                dividend_yield *= 100
        result['Dividend Yield %'] = dividend_yield
        result['52W High'] = safe_info_value(info, 'fiftyTwoWeekHigh')
        result['52W Low'] = safe_info_value(info, 'fiftyTwoWeekLow')
        current_price = result['Current Price']
        if pd.notna(current_price) and pd.notna(result['52W High']) and (result['52W High'] != 0):
            result['Distance From 52W High %'] = (current_price - result['52W High']) / result['52W High'] * 100
        if pd.notna(current_price) and pd.notna(result['52W Low']) and (result['52W Low'] != 0):
            result['Distance From 52W Low %'] = (current_price - result['52W Low']) / result['52W Low'] * 100
        valuation_fields = [result['Trailing P/E'], result['Price / Book'], result['Enterprise Value']]
        if any((pd.notna(x) for x in valuation_fields)):
            result['Valuation Status'] = 'AVAILABLE'
    except:
        pass
    try:
        # Yahoo Finance exposes quarterly statements as date-labelled columns.
        # IMPORTANT: do not trust the generic "Normalized EBITDA" row blindly.
        # Some Yahoo/yfinance records can contain a unit/outlier mismatch. We
        # validate the reported EBITDA against Revenue and, when needed, derive
        # EBITDA from Yahoo's Operating Income + Depreciation & Amortization.
        income = ticker.quarterly_income_stmt
        income_columns = get_sorted_columns(income)
        selected_income_columns = income_columns[:5]
        if len(selected_income_columns) > 0:
            result['Quarterly Status'] = 'AVAILABLE'

        for i, (date, column) in enumerate(selected_income_columns[:4]):
            if i == 0:
                result['Latest Quarter Date'] = date.strftime('%Y-%m-%d')
            elif i == 1:
                result['Previous Quarter Date'] = date.strftime('%Y-%m-%d')
            elif i == 2:
                result['Quarter -3 Date'] = date.strftime('%Y-%m-%d')
            elif i == 3:
                result['Quarter -4 Date'] = date.strftime('%Y-%m-%d')

        revenue_row = safe_row(income, ['Total Revenue', 'Operating Revenue', 'Revenue'])
        profit_row = safe_row(income, ['Net Income', 'Net Income Common Stockholders', 'Net Income Including Noncontrolling Interests'])
        reported_ebitda_row = safe_row(income, ['EBITDA'])
        normalized_ebitda_row = safe_row(income, ['Normalized EBITDA'])
        operating_income_row = safe_row(income, ['Operating Income', 'Operating Income Or Loss'])
        da_row = safe_row(income, [
            'Depreciation And Amortization',
            'Depreciation And Amortization In Income Statement',
            'Reconciled Depreciation',
            'Depreciation',
        ])

        revenue = []
        profit = []
        ebitda = []

        def row_value(series, column):
            if series is None:
                return np.nan
            try:
                return safe_number(series[column])
            except Exception:
                return np.nan

        for i, (_, column) in enumerate(selected_income_columns[:4]):
            q = 4 - i
            revenue_value = row_value(revenue_row, column)
            profit_value = row_value(profit_row, column)
            reported_ebitda = row_value(reported_ebitda_row, column)
            normalized_ebitda = row_value(normalized_ebitda_row, column)
            operating_income = row_value(operating_income_row, column)
            depreciation = row_value(da_row, column)

            # Prefer the explicit Yahoo EBITDA row when available. If Yahoo's
            # reported/normalized EBITDA is wildly larger than revenue, treat it
            # as a likely data/unit anomaly and use a transparent derived value.
            direct_ebitda = reported_ebitda if pd.notna(reported_ebitda) else normalized_ebitda
            derived_ebitda = np.nan
            if pd.notna(operating_income) and pd.notna(depreciation):
                derived_ebitda = operating_income + depreciation

            direct_plausible = pd.notna(direct_ebitda)
            if direct_plausible and pd.notna(revenue_value) and revenue_value != 0:
                # A positive/negative EBITDA can legitimately differ from revenue,
                # but a multi-fold outlier is a strong data-quality warning.
                direct_plausible = abs(direct_ebitda) <= abs(revenue_value) * 2.0

            if direct_plausible:
                ebitda_value = direct_ebitda
                source = 'Yahoo EBITDA' if pd.notna(reported_ebitda) else 'Yahoo Normalized EBITDA'
            elif pd.notna(derived_ebitda):
                ebitda_value = derived_ebitda
                source = 'Derived: Operating Income + D&A'
            else:
                ebitda_value = direct_ebitda
                source = 'Yahoo EBITDA (unvalidated)' if pd.notna(reported_ebitda) else 'Yahoo Normalized EBITDA (unvalidated)' if pd.notna(normalized_ebitda) else 'Unavailable'

            revenue.append(revenue_value)
            profit.append(profit_value)
            ebitda.append(ebitda_value)
            result[f'Revenue Q{q}'] = revenue_value
            result[f'Profit Q{q}'] = profit_value
            result[f'EBITDA Q{q}'] = ebitda_value
            result[f'EBITDA Q{q} Source'] = source
            result[f'Yahoo EBITDA Q{q}'] = direct_ebitda
            result[f'Derived EBITDA Q{q}'] = derived_ebitda

        # Fifth statement column is used for the latest-quarter YoY comparison.
        if len(selected_income_columns) >= 5:
            fifth_column = selected_income_columns[4][1]
            result['Revenue Q4 YoY %'] = safe_pct_change(revenue[0] if revenue else np.nan, row_value(revenue_row, fifth_column))
            result['Profit Q4 YoY %'] = safe_pct_change(profit[0] if profit else np.nan, row_value(profit_row, fifth_column))
            fifth_reported = row_value(reported_ebitda_row, fifth_column)
            fifth_normalized = row_value(normalized_ebitda_row, fifth_column)
            fifth_derived = np.nan
            fifth_revenue = row_value(revenue_row, fifth_column)
            fifth_operating = row_value(operating_income_row, fifth_column)
            fifth_da = row_value(da_row, fifth_column)
            if pd.notna(fifth_operating) and pd.notna(fifth_da):
                fifth_derived = fifth_operating + fifth_da
            fifth_direct = fifth_reported if pd.notna(fifth_reported) else fifth_normalized
            fifth_plausible = pd.notna(fifth_direct)
            if fifth_plausible and pd.notna(fifth_revenue) and fifth_revenue != 0:
                fifth_plausible = abs(fifth_direct) <= abs(fifth_revenue) * 2.0
            fifth_ebitda = fifth_direct if fifth_plausible else fifth_derived if pd.notna(fifth_derived) else fifth_direct
            result['EBITDA Q4 YoY %'] = safe_pct_change(ebitda[0] if ebitda else np.nan, fifth_ebitda)

        result['Revenue Q4 vs Q3 %'] = safe_pct_change(revenue[0] if len(revenue) > 0 else np.nan, revenue[1] if len(revenue) > 1 else np.nan)
        result['Revenue Q3 vs Q2 %'] = safe_pct_change(revenue[1] if len(revenue) > 1 else np.nan, revenue[2] if len(revenue) > 2 else np.nan)
        result['Revenue Q2 vs Q1 %'] = safe_pct_change(revenue[2] if len(revenue) > 2 else np.nan, revenue[3] if len(revenue) > 3 else np.nan)
        result['Profit Q4 vs Q3 %'] = safe_pct_change(profit[0] if len(profit) > 0 else np.nan, profit[1] if len(profit) > 1 else np.nan)
        result['Profit Q3 vs Q2 %'] = safe_pct_change(profit[1] if len(profit) > 1 else np.nan, profit[2] if len(profit) > 2 else np.nan)
        result['Profit Q2 vs Q1 %'] = safe_pct_change(profit[2] if len(profit) > 2 else np.nan, profit[3] if len(profit) > 3 else np.nan)
        result['EBITDA Q4 vs Q3 %'] = safe_pct_change(ebitda[0] if len(ebitda) > 0 else np.nan, ebitda[1] if len(ebitda) > 1 else np.nan)
        result['EBITDA Q3 vs Q2 %'] = safe_pct_change(ebitda[1] if len(ebitda) > 1 else np.nan, ebitda[2] if len(ebitda) > 2 else np.nan)
        result['EBITDA Q2 vs Q1 %'] = safe_pct_change(ebitda[2] if len(ebitda) > 2 else np.nan, ebitda[3] if len(ebitda) > 3 else np.nan)

        net_margins = []
        ebitda_margins = []
        for i in range(4):
            rev = revenue[i] if i < len(revenue) else np.nan
            prof = profit[i] if i < len(profit) else np.nan
            ebit = ebitda[i] if i < len(ebitda) else np.nan
            net_margin = safe_divide(prof, rev) * 100 if pd.notna(safe_divide(prof, rev)) else np.nan
            ebitda_margin = safe_divide(ebit, rev) * 100 if pd.notna(safe_divide(ebit, rev)) else np.nan
            net_margins.append(net_margin)
            ebitda_margins.append(ebitda_margin)
            result[f'Net Margin Q{4 - i} %'] = net_margin
            result[f'EBITDA Margin Q{4 - i} %'] = ebitda_margin
        if len(net_margins) >= 2 and pd.notna(net_margins[0]) and pd.notna(net_margins[1]):
            result['Net Margin Q4 vs Q3 Change'] = net_margins[0] - net_margins[1]
        if len(ebitda_margins) >= 2 and pd.notna(ebitda_margins[0]) and pd.notna(ebitda_margins[1]):
            result['EBITDA Margin Q4 vs Q3 Change'] = ebitda_margins[0] - ebitda_margins[1]
        result['Revenue Trend'] = calculate_trend(revenue[::-1])
        result['Profit Trend'] = calculate_trend(profit[::-1])
        result['EBITDA Trend'] = calculate_trend(ebitda[::-1])
    except Exception:
        pass
    try:
        balance = ticker.quarterly_balance_sheet
        balance_columns = get_sorted_columns(balance)
        selected_balance_columns = balance_columns[:4]
        debt_row = safe_row(balance, ['Total Debt', 'TotalDebt'])
        debt = []
        for i, (date, column) in enumerate(selected_balance_columns):
            value = np.nan
            if debt_row is not None:
                try:
                    value = safe_number(debt_row[column])
                except:
                    pass
            debt.append(value)
            result[f'Debt Q{4 - i}'] = value
        cash_row = safe_row(balance, ['Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents', 'Cash Financial', 'Cash'])
        cash = []
        for i, (date, column) in enumerate(selected_balance_columns):
            value = np.nan
            if cash_row is not None:
                try:
                    value = safe_number(cash_row[column])
                except:
                    pass
            cash.append(value)
            result[f'Cash Q{4 - i}'] = value
        result['Debt Q4 vs Q3 %'] = safe_pct_change(debt[0] if len(debt) > 0 else np.nan, debt[1] if len(debt) > 1 else np.nan)
        result['Debt Q3 vs Q2 %'] = safe_pct_change(debt[1] if len(debt) > 1 else np.nan, debt[2] if len(debt) > 2 else np.nan)
        result['Debt Q2 vs Q1 %'] = safe_pct_change(debt[2] if len(debt) > 2 else np.nan, debt[3] if len(debt) > 3 else np.nan)
        result['Current Debt'] = debt[0] if len(debt) > 0 else np.nan
        result['Previous Quarter Debt'] = debt[1] if len(debt) > 1 else np.nan
        result['Debt Change %'] = safe_pct_change(result['Current Debt'], result['Previous Quarter Debt'])
        if pd.notna(result['Current Debt']):
            result['Debt Status'] = 'AVAILABLE'
        elif len(debt) > 0:
            result['Debt Status'] = 'CURRENT_ONLY'
        result['Debt Trend'] = calculate_trend(debt[::-1])
        result['Cash Q4 vs Q3 %'] = safe_pct_change(cash[0] if len(cash) > 0 else np.nan, cash[1] if len(cash) > 1 else np.nan)
        result['Cash Trend'] = calculate_trend(cash[::-1])
        net_debt = []
        for i in range(max(len(debt), len(cash))):
            d = debt[i] if i < len(debt) else np.nan
            c = cash[i] if i < len(cash) else np.nan
            if pd.notna(d) and pd.notna(c):
                nd = d - c
            else:
                nd = np.nan
            net_debt.append(nd)
            result[f'Net Debt Q{4 - i}'] = nd
        result['Net Debt Q4 vs Q3 %'] = safe_pct_change(net_debt[0] if len(net_debt) > 0 else np.nan, net_debt[1] if len(net_debt) > 1 else np.nan)
        result['Net Debt Trend'] = calculate_trend(net_debt[::-1])
    except:
        pass
    try:
        cashflow = ticker.quarterly_cashflow
        cashflow_columns = get_sorted_columns(cashflow)
        selected_cashflow_columns = cashflow_columns[:4]
        ocf_row = safe_row(cashflow, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
        ocf = []
        for i, (date, column) in enumerate(selected_cashflow_columns):
            value = np.nan
            if ocf_row is not None:
                try:
                    value = safe_number(ocf_row[column])
                except:
                    pass
            ocf.append(value)
            result[f'Operating CF Q{4 - i}'] = value
        result['Operating CF Q4 vs Q3 %'] = safe_pct_change(ocf[0] if len(ocf) > 0 else np.nan, ocf[1] if len(ocf) > 1 else np.nan)
        result['Operating CF Trend'] = calculate_trend(ocf[::-1])
        capex_row = safe_row(cashflow, ['Capital Expenditure', 'Capital Expenditures'])
        fcf = []
        for i, (date, column) in enumerate(selected_cashflow_columns):
            operating_cf = ocf[i] if i < len(ocf) else np.nan
            capex = np.nan
            if capex_row is not None:
                try:
                    capex = safe_number(capex_row[column])
                except:
                    pass
            if pd.notna(operating_cf) and pd.notna(capex):
                free_cash_flow = operating_cf + capex
            else:
                free_cash_flow = np.nan
            fcf.append(free_cash_flow)
            result[f'FCF Q{4 - i}'] = free_cash_flow
        result['FCF Q4 vs Q3 %'] = safe_pct_change(fcf[0] if len(fcf) > 0 else np.nan, fcf[1] if len(fcf) > 1 else np.nan)
        result['FCF Trend'] = calculate_trend(fcf[::-1])
    except:
        pass
    try:
        earnings_dates = ticker.get_earnings_dates(limit=12)
        if earnings_dates is not None and (not earnings_dates.empty):
            future_dates = []
            current_time = pd.Timestamp.now(tz='UTC')
            for date in earnings_dates.index:
                try:
                    dt = pd.Timestamp(date)
                    if dt.tzinfo is None:
                        dt = dt.tz_localize('UTC')
                    if dt > current_time:
                        future_dates.append(dt)
                except:
                    continue
            if len(future_dates) > 0:
                future_dates = sorted(future_dates)
                result['Next Result Date'] = future_dates[0].strftime('%Y-%m-%d')
                result['Result Date Status'] = 'UPCOMING'
            else:
                result['Result Date Status'] = 'NO_UPCOMING_DATE'
    except:
        result['Result Date Status'] = 'NOT_AVAILABLE'
    # ============================================================
    # INTERPRETABLE MARKET + FUNDAMENTAL ANALYSIS
    # Descriptive evidence only — no BUY/SELL score or recommendation.
    # ============================================================
    try:
        hist_for_trend = ticker.history(period='1y', auto_adjust=False)
        if hist_for_trend is not None and not hist_for_trend.empty:
            close = pd.to_numeric(hist_for_trend['Close'], errors='coerce').dropna()
            if len(close) >= 20:
                dma20 = close.rolling(20).mean().iloc[-1]
                result['20 DMA'] = safe_number(dma20)
                result['Price vs 20 DMA %'] = safe_pct_change(close.iloc[-1], dma20)
            if len(close) >= 50:
                dma50 = close.rolling(50).mean().iloc[-1]
                result['50 DMA'] = safe_number(dma50)
                result['Price vs 50 DMA %'] = safe_pct_change(close.iloc[-1], dma50)
            if len(close) >= 200:
                dma200 = close.rolling(200).mean().iloc[-1]
                result['200 DMA'] = safe_number(dma200)
                result['Price vs 200 DMA %'] = safe_pct_change(close.iloc[-1], dma200)

            p20 = result.get('20 DMA', np.nan); p50 = result.get('50 DMA', np.nan); p200 = result.get('200 DMA', np.nan)
            if all(pd.notna(x) for x in [p20, p50, p200]):
                if p20 > p50 > p200:
                    result['DMA Alignment'] = '20D > 50D > 200D'
                elif p20 < p50 < p200:
                    result['DMA Alignment'] = '20D < 50D < 200D'
                else:
                    result['DMA Alignment'] = 'MIXED'
            else:
                result['DMA Alignment'] = 'PARTIAL'

            p = close.iloc[-1]
            above = []
            below = []
            for label, key in [('20D','20 DMA'),('50D','50 DMA'),('200D','200 DMA')]:
                v = result.get(key, np.nan)
                if pd.notna(v):
                    (above if p > v else below).append(label)
            r1m = np.nan; r3m = np.nan; r6m = np.nan
            for days, name in [(30,'1M'),(90,'3M'),(180,'6M')]:
                target = close.index[-1] - pd.Timedelta(days=days)
                prior = close.loc[:target]
                val = np.nan if prior.empty or prior.iloc[-1] == 0 else (p/prior.iloc[-1]-1)*100
                result[f'{name} Trend Return %'] = val
                if name == '1M': r1m = val
                elif name == '3M': r3m = val
                else: r6m = val
            valid_returns = [x for x in [r1m,r3m,r6m] if pd.notna(x)]
            if len(valid_returns) >= 2:
                pos = sum(x > 0 for x in valid_returns); neg = sum(x < 0 for x in valid_returns)
                result['Momentum Direction'] = 'POSITIVE' if pos > neg else 'NEGATIVE' if neg > pos else 'MIXED'
            else:
                result['Momentum Direction'] = 'NO_DATA'
            if above and len(above) >= len(below) + 1:
                result['Market Trend'] = 'UPWARD'
            elif below and len(below) >= len(above) + 1:
                result['Market Trend'] = 'DOWNWARD'
            else:
                result['Market Trend'] = 'MIXED'
            evidence=[]
            if above: evidence.append('Price above ' + ', '.join(above) + ' DMA')
            if below: evidence.append('Price below ' + ', '.join(below) + ' DMA')
            if result.get('DMA Alignment') in ('20D > 50D > 200D','20D < 50D < 200D'):
                evidence.append('DMA alignment ' + result['DMA Alignment'])
            result['Market Trend Evidence'] = '; '.join(evidence) if evidence else 'Insufficient DMA data'

            vol = pd.to_numeric(hist_for_trend.get('Volume', pd.Series(dtype=float)), errors='coerce').dropna()
            if len(vol) >= 20 and vol.iloc[-1] != 0:
                avg20 = vol.tail(20).mean()
                result['Volume vs 20D Avg %'] = (vol.iloc[-1]/avg20 - 1)*100 if avg20 else np.nan
    except Exception:
        result['Market Trend'] = result.get('Market Trend','NO_DATA')
        result['Market Trend Evidence'] = result.get('Market Trend Evidence','Insufficient market data')

    # Descriptive financial directions.
    result['Revenue Direction'] = result.get('Revenue Trend', 'NO_DATA')
    result['Profit Direction'] = result.get('Profit Trend', 'NO_DATA')
    result['EBITDA Direction'] = result.get('EBITDA Trend', 'NO_DATA')
    result['Debt Direction'] = result.get('Debt Trend', 'NO_DATA')
    result['Cash Direction'] = result.get('Cash Trend', 'NO_DATA')
    result['Net Debt Direction'] = result.get('Net Debt Trend', 'NO_DATA')
    result['Operating CF Direction'] = result.get('Operating CF Trend', 'NO_DATA')
    result['FCF Direction'] = result.get('FCF Trend', 'NO_DATA')

    positive=[]
    negative=[]
    def add_signal(target, text):
        if text and text not in target:
            target.append(text)
    if result.get('Revenue Q4 vs Q3 %') is not None and pd.notna(result.get('Revenue Q4 vs Q3 %')):
        v=result['Revenue Q4 vs Q3 %']; add_signal(positive, f'Revenue increased QoQ by {v:.1f}%') if v>0 else add_signal(negative, f'Revenue decreased QoQ by {abs(v):.1f}%') if v<0 else None
    if result.get('Revenue Q4 YoY %') is not None and pd.notna(result.get('Revenue Q4 YoY %')):
        v=result['Revenue Q4 YoY %']; add_signal(positive, f'Revenue increased YoY by {v:.1f}%') if v>0 else add_signal(negative, f'Revenue decreased YoY by {abs(v):.1f}%') if v<0 else None
    if result.get('Profit Q4 vs Q3 %') is not None and pd.notna(result.get('Profit Q4 vs Q3 %')):
        v=result['Profit Q4 vs Q3 %']; add_signal(positive, f'Profit increased QoQ by {v:.1f}%') if v>0 else add_signal(negative, f'Profit decreased QoQ by {abs(v):.1f}%') if v<0 else None
    if result.get('Profit Q4 YoY %') is not None and pd.notna(result.get('Profit Q4 YoY %')):
        v=result['Profit Q4 YoY %']; add_signal(positive, f'Profit increased YoY by {v:.1f}%') if v>0 else add_signal(negative, f'Profit decreased YoY by {abs(v):.1f}%') if v<0 else None
    if result.get('Net Margin Q4 vs Q3 Change') is not None and pd.notna(result.get('Net Margin Q4 vs Q3 Change')):
        v=result['Net Margin Q4 vs Q3 Change']; add_signal(positive, f'Net margin expanded by {v:.2f} percentage points QoQ') if v>0 else add_signal(negative, f'Net margin contracted by {abs(v):.2f} percentage points QoQ') if v<0 else None
    if result.get('Debt Q4 vs Q3 %') is not None and pd.notna(result.get('Debt Q4 vs Q3 %')):
        v=result['Debt Q4 vs Q3 %']; add_signal(positive, f'Debt decreased QoQ by {abs(v):.1f}%') if v<0 else add_signal(negative, f'Debt increased QoQ by {v:.1f}%') if v>0 else None
    if result.get('Net Debt Q4 vs Q3 %') is not None and pd.notna(result.get('Net Debt Q4 vs Q3 %')):
        v=result['Net Debt Q4 vs Q3 %']; add_signal(positive, f'Net debt decreased QoQ by {abs(v):.1f}%') if v<0 else add_signal(negative, f'Net debt increased QoQ by {v:.1f}%') if v>0 else None
    if result.get('Operating CF Q4') is not None and pd.notna(result.get('Operating CF Q4')):
        add_signal(positive, 'Latest-quarter operating cash flow is positive') if result['Operating CF Q4'] > 0 else add_signal(negative, 'Latest-quarter operating cash flow is negative')
    if result.get('FCF Q4') is not None and pd.notna(result.get('FCF Q4')):
        add_signal(positive, 'Latest-quarter free cash flow is positive') if result['FCF Q4'] > 0 else add_signal(negative, 'Latest-quarter free cash flow is negative')
    if result.get('Market Trend') == 'UPWARD': add_signal(positive, 'Price structure is upward based on available DMA evidence')
    elif result.get('Market Trend') == 'DOWNWARD': add_signal(negative, 'Price structure is downward based on available DMA evidence')
    elif result.get('Market Trend') == 'MIXED': add_signal(negative, 'Price structure is mixed across the available DMA measures')
    if result.get('Momentum Direction') == 'POSITIVE': add_signal(positive, '1M/3M/6M returns are positive in the majority of available periods')
    elif result.get('Momentum Direction') == 'NEGATIVE': add_signal(negative, '1M/3M/6M returns are negative in the majority of available periods')
    result['Positive Evidence'] = ' | '.join(positive) if positive else 'No clear positive change identified from available fields'
    result['Negative / Watch Evidence'] = ' | '.join(negative) if negative else 'No clear negative change identified from available fields'

    important_fields = ['Current Price', 'Market Cap', 'Revenue Q4', 'Profit Q4', 'EBITDA Q4', 'Current Debt', 'Cash Q4', 'Operating CF Q4', 'FCF Q4']
    available_count = 0
    for field in important_fields:
        if pd.notna(result.get(field, np.nan)):
            available_count += 1
    if available_count >= 7:
        result['Overall Data Status'] = 'GOOD'
    elif available_count >= 3:
        result['Overall Data Status'] = 'PARTIAL'
    else:
        result['Overall Data Status'] = 'NO_DATA'
    return result
