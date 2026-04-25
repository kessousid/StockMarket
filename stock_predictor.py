"""
Stock Market Analysis Tool (India + US)
Provides Bullish/Neutral/Bearish signals for NSE and US stocks using technical analysis,
news sentiment analysis, and quarterly fundamental analysis.
FOR EDUCATIONAL PURPOSES ONLY. NOT INVESTMENT ADVICE.
"""

# =============================================================================
# Section 1: Imports & Constants
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os

# Heavy imports deferred to first use so Railway serves the first page faster
def _yf():
    import yfinance as yf
    return yf

def _feedparser():
    import feedparser
    return feedparser

def _go():
    import plotly.graph_objects as go
    return go

def _make_subplots():
    from plotly.subplots import make_subplots
    return make_subplots

@st.cache_resource(show_spinner=False)
def _get_vader_analyzer():
    """Initialize VADER once per process; lexicon load is expensive on cold start."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

# Technical analysis constants
SMA_SHORT = 20
SMA_LONG = 50
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL_PERIOD = 9
BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
VOLUME_MA_PERIOD = 20

# Technical component weights (must sum to 1)
TECH_SMA_WEIGHT = 0.18
TECH_RSI_WEIGHT = 0.18
TECH_MACD_WEIGHT = 0.20
TECH_BB_WEIGHT = 0.15
TECH_OBV_WEIGHT = 0.14
TECH_MOMENTUM_WEIGHT = 0.15

SENTIMENT_STOCK_WEIGHT = 0.50
SENTIMENT_SECTOR_WEIGHT = 0.30
SENTIMENT_MARKET_WEIGHT = 0.20

FUND_REVENUE_WEIGHT = 0.13
FUND_MARGIN_WEIGHT = 0.10
FUND_PROFIT_WEIGHT = 0.13
FUND_DEBT_EQUITY_WEIGHT = 0.15
FUND_CURRENT_RATIO_WEIGHT = 0.09
FUND_ROE_WEIGHT = 0.15
FUND_ROCA_WEIGHT = 0.12
FUND_ROCE_WEIGHT = 0.13

# Final prediction weights
WEIGHT_TECHNICAL = 0.40
WEIGHT_FUNDAMENTAL = 0.35
WEIGHT_SENTIMENT = 0.25

# Prediction thresholds
BUY_THRESHOLD = 0.3
SELL_THRESHOLD = -0.3

# =============================================================================
# Section 2: Stock Dictionaries (by Market Cap Category)
# =============================================================================

NIFTY_50 = {
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "BPCL": "BPCL.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Britannia": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Dr. Reddy's": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Grasim Industries": "GRASIM.NS",
    "HCL Technologies": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco": "HINDALCO.NS",
    "HUL": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Infosys": "INFY.NS",
    "ITC": "ITC.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "L&T": "LT.NS",
    "M&M": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Power Grid": "POWERGRID.NS",
    "Reliance Industries": "RELIANCE.NS",
    "SBI": "SBIN.NS",
    "SBI Life": "SBILIFE.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "TCS": "TCS.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan": "TITAN.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "UPL": "UPL.NS",
    "Wipro": "WIPRO.NS",
}

NIFTY_NEXT_50 = {
    "Adani Green": "ADANIGREEN.NS",
    "Adani Total Gas": "ATGL.NS",
    "Ambuja Cements": "AMBUJACEM.NS",
    "Ashok Leyland": "ASHOKLEY.NS",
    "Avenue Supermarts": "DMART.NS",
    "Bajaj Holdings": "BAJAJHLDNG.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "Berger Paints": "BERGEPAINT.NS",
    "Biocon": "BIOCON.NS",
    "Bosch": "BOSCHLTD.NS",
    "Canara Bank": "CANBK.NS",
    "Cholamandalam": "CHOLAFIN.NS",
    "Colgate": "COLPAL.NS",
    "Dabur": "DABUR.NS",
    "DLF": "DLF.NS",
    "Godrej Consumer": "GODREJCP.NS",
    "Havells": "HAVELLS.NS",
    "ICICI Lombard": "ICICIGI.NS",
    "ICICI Prudential": "ICICIPRULI.NS",
    "IDFC First Bank": "IDFCFIRSTB.NS",
    "Indian Hotels": "INDHOTEL.NS",
    "Indus Towers": "INDUSTOWER.NS",
    "Info Edge": "NAUKRI.NS",
    "InterGlobe Aviation": "INDIGO.NS",
    "IOC": "IOC.NS",
    "Jindal Steel": "JINDALSTEL.NS",
    "LIC": "LICI.NS",
    "Lupin": "LUPIN.NS",
    "Marico": "MARICO.NS",
    "Max Healthcare": "MAXHEALTH.NS",
    "Motherson Sumi": "MOTHERSON.NS",
    "NHPC": "NHPC.NS",
    "Pidilite": "PIDILITIND.NS",
    "PNB": "PNB.NS",
    "PFC": "PFC.NS",
    "REC": "RECLTD.NS",
    "SBI Cards": "SBICARD.NS",
    "Shree Cement": "SHREECEM.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Siemens": "SIEMENS.NS",
    "SRF": "SRF.NS",
    "Tata Communications": "TATACOMM.NS",
    "Tata Elxsi": "TATAELXSI.NS",
    "Tata Power": "TATAPOWER.NS",
    "Torrent Pharma": "TORNTPHARM.NS",
    "TVS Motor": "TVSMOTOR.NS",
    "United Spirits": "UNITDSPR.NS",
    "Vedanta": "VEDL.NS",
    "Zomato": "ZOMATO.NS",
    "Zydus Lifesciences": "ZYDUSLIFE.NS",
}

MIDCAP_STOCKS = {
    "Aarti Industries": "AARTIIND.NS",
    "ABB India": "ABB.NS",
    "ACC": "ACC.NS",
    "Aditya Birla Fashion": "ABFRL.NS",
    "Alkem Laboratories": "ALKEM.NS",
    "Astral": "ASTRAL.NS",
    "AU Small Finance Bank": "AUBANK.NS",
    "Balkrishna Industries": "BALKRISIND.NS",
    "Bandhan Bank": "BANDHANBNK.NS",
    "Bata India": "BATAINDIA.NS",
    "BEL": "BEL.NS",
    "Bharat Forge": "BHARATFORG.NS",
    "BSE": "BSE.NS",
    "Canfin Homes": "CANFINHOME.NS",
    "Carborundum Universal": "CARBORUNIV.NS",
    "CDSL": "CDSL.NS",
    "Central Bank": "CENTRALBK.NS",
    "Chalet Hotels": "CHALET.NS",
    "City Union Bank": "CUB.NS",
    "Clean Science": "CLEAN.NS",
    "Coforge": "COFORGE.NS",
    "CG Power": "CGPOWER.NS",
    "Crompton Greaves": "CROMPTON.NS",
    "Cummins India": "CUMMINSIND.NS",
    "Deepak Nitrite": "DEEPAKNTR.NS",
    "Delhivery": "DELHIVERY.NS",
    "Dixon Technologies": "DIXON.NS",
    "Emami": "EMAMILTD.NS",
    "Endurance Technologies": "ENDURANCE.NS",
    "Escorts Kubota": "ESCORTS.NS",
    "Federal Bank": "FEDERALBNK.NS",
    "Fortis Healthcare": "FORTIS.NS",
    "GAIL": "GAIL.NS",
    "Gillette India": "GILLETTE.NS",
    "Glaxo Pharma": "GLAXO.NS",
    "GMR Airports": "GMRINFRA.NS",
    "Godrej Properties": "GODREJPROP.NS",
    "Gujarat Gas": "GUJGASLTD.NS",
    "Gujarat Fluorochemicals": "FLUOROCHEM.NS",
    "HAL": "HAL.NS",
    "HDFC AMC": "HDFCAMC.NS",
    "Honeywell Automation": "HONAUT.NS",
    "IDBI Bank": "IDBI.NS",
    "IPCL": "IPCALAB.NS",
    "Indian Energy Exchange": "IEX.NS",
    "IRCTC": "IRCTC.NS",
    "JK Cement": "JKCEMENT.NS",
    "JSW Energy": "JSWENERGY.NS",
    "Jubilant FoodWorks": "JUBLFOOD.NS",
    "Kajaria Ceramics": "KAJARIACER.NS",
    "KEI Industries": "KEI.NS",
    "KPIT Technologies": "KPITTECH.NS",
    "L&T Finance": "LTF.NS",
    "L&T Technology": "LTTS.NS",
    "Laurus Labs": "LAURUSLABS.NS",
    "Manappuram Finance": "MANAPPURAM.NS",
    "Mphasis": "MPHASIS.NS",
    "MRF": "MRF.NS",
    "Muthoot Finance": "MUTHOOTFIN.NS",
    "Narayana Hrudayalaya": "NH.NS",
    "National Aluminium": "NATIONALUM.NS",
    "Nippon Life AMC": "NAM-INDIA.NS",
    "Nykaa": "NYKAA.NS",
    "Oberoi Realty": "OBEROIRLTY.NS",
    "Oracle Financial": "OFSS.NS",
    "Page Industries": "PAGEIND.NS",
    "Paytm": "PAYTM.NS",
    "Persistent Systems": "PERSISTENT.NS",
    "Petronet LNG": "PETRONET.NS",
    "PI Industries": "PIIND.NS",
    "Polycab India": "POLYCAB.NS",
    "Prestige Estates": "PRESTIGE.NS",
    "PVR Inox": "PVRINOX.NS",
    "Radico Khaitan": "RADICO.NS",
    "Rajesh Exports": "RAJESHEXPO.NS",
    "Rashtriya Chemicals": "RCF.NS",
    "Sail": "SAIL.NS",
    "Sona BLW": "SONACOMS.NS",
    "Sundaram Finance": "SUNDARMFIN.NS",
    "Supreme Industries": "SUPREMEIND.NS",
    "Syngene International": "SYNGENE.NS",
    "Tata Chemicals": "TATACHEM.NS",
    "Tata Technologies": "TATATECH.NS",
    "Thermax": "THERMAX.NS",
    "Timken India": "TIMKEN.NS",
    "Trent": "TRENT.NS",
    "Tube Investments": "TIINDIA.NS",
    "UCO Bank": "UCOBANK.NS",
    "Union Bank": "UNIONBANK.NS",
    "United Breweries": "UBL.NS",
    "Varun Beverages": "VBL.NS",
    "Voltas": "VOLTAS.NS",
    "Whirlpool India": "WHIRLPOOL.NS",
    "Yes Bank": "YESBANK.NS",
    "Zee Entertainment": "ZEEL.NS",
    "Zomato": "ZOMATO.NS",
}

SMALLCAP_STOCKS = {
    "Affle India": "AFFLE.NS",
    "Angel One": "ANGELONE.NS",
    "Birlasoft": "BSOFT.NS",
    "BLS International": "BLS.NS",
    "Campus Activewear": "CAMPUS.NS",
    "Can Fin Homes": "CANFINHOME.NS",
    "Castrol India": "CASTROLIND.NS",
    "Central Depository": "CDSL.NS",
    "Cochin Shipyard": "COCHINSHIP.NS",
    "Computer Age Mgmt": "CAMS.NS",
    "CreditAccess Grameen": "CREDITACC.NS",
    "Cyient": "CYIENT.NS",
    "Data Patterns": "DATAPATTNS.NS",
    "Deepak Fertilisers": "DEEPAKFERT.NS",
    "EaseMyTrip": "EASEMYTRIP.NS",
    "Electronics Mart": "ELECTRONI.NS",
    "Elgi Equipments": "ELGIEQUIP.NS",
    "Engineers India": "ENGINERSIN.NS",
    "Finolex Cables": "FINCABLES.NS",
    "Firstsource Solutions": "FSL.NS",
    "Five Star Finance": "FIVESTAR.NS",
    "Go Digit": "GODIGIT.NS",
    "Gravita India": "GRAVITA.NS",
    "GRSE": "GRSE.NS",
    "Happiest Minds": "HAPPSTMNDS.NS",
    "HBL Power": "HBLPOWER.NS",
    "HFCL": "HFCL.NS",
    "Himadri Speciality": "HSCL.NS",
    "HUDCO": "HUDCO.NS",
    "IIFL Finance": "IIFL.NS",
    "IndiaMart": "INDIAMART.NS",
    "Indian Bank": "INDIANB.NS",
    "IRCON International": "IRCON.NS",
    "IRFC": "IRFC.NS",
    "ITI Limited": "ITI.NS",
    "JBM Auto": "JBMA.NS",
    "Jubilant Ingrevia": "JUBLINGREA.NS",
    "Jupiter Wagons": "JWL.NS",
    "Kalyan Jewellers": "KALYANKJIL.NS",
    "KEC International": "KEC.NS",
    "Kirloskar Brothers": "KIRLOSBROS.NS",
    "KPIT Technologies": "KPITTECH.NS",
    "Lemon Tree Hotels": "LEMONTREE.NS",
    "Lloyds Metals": "LLOYDSMETA.NS",
    "Mazagon Dock": "MAZDOCK.NS",
    "MOIL": "MOIL.NS",
    "Motilal Oswal": "MOTILALOFS.NS",
    "Multi Commodity Exch": "MCX.NS",
    "NBCC": "NBCC.NS",
    "NCC": "NCC.NS",
    "NMDC Steel": "NMDCSTEEL.NS",
    "Olectra Greentech": "OLECTRA.NS",
    "PCBL": "PCBL.NS",
    "PNB Housing Finance": "PNBHOUSING.NS",
    "Poonawalla Fincorp": "POONAWALLA.NS",
    "Quess Corp": "QUESS.NS",
    "RITES": "RITES.NS",
    "RateGain Travel": "RATEGAIN.NS",
    "Raymond": "RAYMOND.NS",
    "Redington": "REDINGTON.NS",
    "Route Mobile": "ROUTE.NS",
    "RVNL": "RVNL.NS",
    "Sapphire Foods": "SAPPHIRE.NS",
    "Senco Gold": "SENCO.NS",
    "Shilpa Medicare": "SHILPAMED.NS",
    "Shyam Metalics": "SHYAMMETL.NS",
    "Sobha": "SOBHA.NS",
    "Solar Industries": "SOLARINDS.NS",
    "Sonata Software": "SONATSOFTW.NS",
    "Star Health": "STARHEALTH.NS",
    "Sterling & Wilson": "SWSOLAR.NS",
    "Sumitomo Chemical": "SUMICHEM.NS",
    "Sundram Fasteners": "SUNDRMFAST.NS",
    "Suzlon Energy": "SUZLON.NS",
    "Tanla Platforms": "TANLA.NS",
    "Tata Elxsi": "TATAELXSI.NS",
    "Tejas Networks": "TEJASNET.NS",
    "Titagarh Rail": "TITAGARH.NS",
    "Torrent Power": "TORNTPOWER.NS",
    "Trident": "TRIDENT.NS",
    "Triveni Turbine": "TRITURBINE.NS",
    "UTI AMC": "UTIAMC.NS",
    "VADILAL": "VADILALIND.NS",
    "Vaibhav Global": "VAIBHAVGBL.NS",
    "Vardhman Textiles": "VTL.NS",
    "Welspun Living": "WELSPUNLIV.NS",
    "Zensar Technologies": "ZENSARTECH.NS",
    "Zomato": "ZOMATO.NS",
}

MICROCAP_STOCKS = {
    "Butterfly Gandhimathi": "BUTTERFLY.NS",
    "CARE Ratings": "CARERATING.NS",
    "Dhampur Sugar": "DHAMPURSUG.NS",
    "Elin Electronics": "ELIN.NS",
    "Gokaldas Exports": "GOKALDAS.NS",
    "GTPL Hathway": "GTPL.NS",
    "Gulf Oil Lubricants": "GULFOILLUB.NS",
    "Inox Wind": "INOXWIND.NS",
    "ISGEC Heavy Engineering": "ISGEC.NS",
    "JK Paper": "JKPAPER.NS",
    "Kopran": "KOPRAN.NS",
    "Lux Industries": "LUXIND.NS",
    "Maithan Alloys": "MAITHANALL.NS",
    "Maharashtra Seamless": "MAHSEAMLES.NS",
    "Midhani": "MIDHANI.NS",
    "MSTC": "MSTC.NS",
    "Nelcast": "NELCAST.NS",
    "Orient Electric": "ORIENTELEC.NS",
    "Patel Engineering": "PATELENG.NS",
    "Peninsula Land": "PENINLAND.NS",
    "RACL Geartech": "RACLGEAR.NS",
    "Rajratan Global Wire": "RAJRATAN.NS",
    "Roto Pumps": "ROTOPUMPS.NS",
    "Stylam Industries": "STYLAM.NS",
    "TCNS Clothing": "TCNSBRANDS.NS",
    "Tinna Rubber": "TINNARUBR.NS",
    "VIP Industries": "VIPIND.NS",
    "West Coast Paper": "WESTCOAST.NS",
    "Wonderla Holidays": "WONDERLA.NS",
    "Zaggle Prepaid": "ZAGGLE.NS",
}

STOCK_CATEGORIES = {
    "Large Cap (Nifty 50)": NIFTY_50,
    "Nifty Next 50": NIFTY_NEXT_50,
    "Mid Cap": MIDCAP_STOCKS,
    "Small Cap": SMALLCAP_STOCKS,
    "Micro Cap": MICROCAP_STOCKS,
}

STOCK_SECTORS = {
    "Banking": {
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "Axis Bank": "AXISBANK.NS",
        "SBI": "SBIN.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS",
        "IndusInd Bank": "INDUSINDBK.NS",
        "Bank of Baroda": "BANKBARODA.NS",
        "Canara Bank": "CANBK.NS",
        "PNB": "PNB.NS",
        "IDFC First Bank": "IDFCFIRSTB.NS",
        "AU Small Finance Bank": "AUBANK.NS",
        "Bandhan Bank": "BANDHANBNK.NS",
        "Federal Bank": "FEDERALBNK.NS",
        "IDBI Bank": "IDBI.NS",
        "Yes Bank": "YESBANK.NS",
        "Central Bank": "CENTRALBK.NS",
        "UCO Bank": "UCOBANK.NS",
        "Union Bank": "UNIONBANK.NS",
        "City Union Bank": "CUB.NS",
        "Indian Bank": "INDIANB.NS",
    },
    "IT / Technology": {
        "TCS": "TCS.NS",
        "Infosys": "INFY.NS",
        "HCL Technologies": "HCLTECH.NS",
        "Wipro": "WIPRO.NS",
        "Tech Mahindra": "TECHM.NS",
        "Tata Elxsi": "TATAELXSI.NS",
        "Info Edge": "NAUKRI.NS",
        "Tata Communications": "TATACOMM.NS",
        "Coforge": "COFORGE.NS",
        "KPIT Technologies": "KPITTECH.NS",
        "L&T Technology": "LTTS.NS",
        "Mphasis": "MPHASIS.NS",
        "Oracle Financial": "OFSS.NS",
        "Persistent Systems": "PERSISTENT.NS",
        "Tata Technologies": "TATATECH.NS",
        "Dixon Technologies": "DIXON.NS",
        "Birlasoft": "BSOFT.NS",
        "Cyient": "CYIENT.NS",
        "Firstsource Solutions": "FSL.NS",
        "Happiest Minds": "HAPPSTMNDS.NS",
        "Sonata Software": "SONATSOFTW.NS",
        "Zensar Technologies": "ZENSARTECH.NS",
        "Affle India": "AFFLE.NS",
        "Route Mobile": "ROUTE.NS",
        "Tanla Platforms": "TANLA.NS",
        "RateGain Travel": "RATEGAIN.NS",
        "Redington": "REDINGTON.NS",
    },
    "Pharma / Healthcare": {
        "Sun Pharma": "SUNPHARMA.NS",
        "Cipla": "CIPLA.NS",
        "Divi's Laboratories": "DIVISLAB.NS",
        "Dr. Reddy's": "DRREDDY.NS",
        "Apollo Hospitals": "APOLLOHOSP.NS",
        "Biocon": "BIOCON.NS",
        "Lupin": "LUPIN.NS",
        "Max Healthcare": "MAXHEALTH.NS",
        "Torrent Pharma": "TORNTPHARM.NS",
        "Zydus Lifesciences": "ZYDUSLIFE.NS",
        "Alkem Laboratories": "ALKEM.NS",
        "Fortis Healthcare": "FORTIS.NS",
        "Glaxo Pharma": "GLAXO.NS",
        "IPCL": "IPCALAB.NS",
        "Laurus Labs": "LAURUSLABS.NS",
        "Narayana Hrudayalaya": "NH.NS",
        "Syngene International": "SYNGENE.NS",
        "Shilpa Medicare": "SHILPAMED.NS",
        "Star Health": "STARHEALTH.NS",
    },
    "Automobile": {
        "Bajaj Auto": "BAJAJ-AUTO.NS",
        "Eicher Motors": "EICHERMOT.NS",
        "Hero MotoCorp": "HEROMOTOCO.NS",
        "M&M": "M&M.NS",
        "Maruti Suzuki": "MARUTI.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "Ashok Leyland": "ASHOKLEY.NS",
        "Bosch": "BOSCHLTD.NS",
        "TVS Motor": "TVSMOTOR.NS",
        "Motherson Sumi": "MOTHERSON.NS",
        "Balkrishna Industries": "BALKRISIND.NS",
        "Bharat Forge": "BHARATFORG.NS",
        "Endurance Technologies": "ENDURANCE.NS",
        "Escorts Kubota": "ESCORTS.NS",
        "MRF": "MRF.NS",
        "Sona BLW": "SONACOMS.NS",
        "Tube Investments": "TIINDIA.NS",
        "JBM Auto": "JBMA.NS",
        "Sundram Fasteners": "SUNDRMFAST.NS",
        "Olectra Greentech": "OLECTRA.NS",
    },
    "FMCG / Consumer": {
        "HUL": "HINDUNILVR.NS",
        "ITC": "ITC.NS",
        "Britannia": "BRITANNIA.NS",
        "Nestle India": "NESTLEIND.NS",
        "Tata Consumer": "TATACONSUM.NS",
        "Asian Paints": "ASIANPAINT.NS",
        "Colgate": "COLPAL.NS",
        "Dabur": "DABUR.NS",
        "Godrej Consumer": "GODREJCP.NS",
        "Marico": "MARICO.NS",
        "Berger Paints": "BERGEPAINT.NS",
        "Pidilite": "PIDILITIND.NS",
        "United Spirits": "UNITDSPR.NS",
        "Avenue Supermarts": "DMART.NS",
        "Bata India": "BATAINDIA.NS",
        "Emami": "EMAMILTD.NS",
        "Gillette India": "GILLETTE.NS",
        "Page Industries": "PAGEIND.NS",
        "Radico Khaitan": "RADICO.NS",
        "United Breweries": "UBL.NS",
        "Varun Beverages": "VBL.NS",
        "Aditya Birla Fashion": "ABFRL.NS",
        "Campus Activewear": "CAMPUS.NS",
        "Crompton Greaves": "CROMPTON.NS",
        "Havells": "HAVELLS.NS",
        "Whirlpool India": "WHIRLPOOL.NS",
        "Trent": "TRENT.NS",
        "Titan": "TITAN.NS",
        "Kalyan Jewellers": "KALYANKJIL.NS",
        "Senco Gold": "SENCO.NS",
        "Raymond": "RAYMOND.NS",
        "VADILAL": "VADILALIND.NS",
        "Sapphire Foods": "SAPPHIRE.NS",
        "Jubilant FoodWorks": "JUBLFOOD.NS",
        "Nykaa": "NYKAA.NS",
        "Electronics Mart": "ELECTRONI.NS",
    },
    "Oil & Gas / Energy": {
        "Reliance Industries": "RELIANCE.NS",
        "BPCL": "BPCL.NS",
        "ONGC": "ONGC.NS",
        "IOC": "IOC.NS",
        "Adani Total Gas": "ATGL.NS",
        "Adani Green": "ADANIGREEN.NS",
        "GAIL": "GAIL.NS",
        "Gujarat Gas": "GUJGASLTD.NS",
        "Petronet LNG": "PETRONET.NS",
        "Castrol India": "CASTROLIND.NS",
        "Tata Power": "TATAPOWER.NS",
        "JSW Energy": "JSWENERGY.NS",
        "NHPC": "NHPC.NS",
        "NTPC": "NTPC.NS",
        "Power Grid": "POWERGRID.NS",
        "Torrent Power": "TORNTPOWER.NS",
        "Adani Enterprises": "ADANIENT.NS",
        "Suzlon Energy": "SUZLON.NS",
        "Sterling & Wilson": "SWSOLAR.NS",
        "Coal India": "COALINDIA.NS",
    },
    "Finance / Insurance": {
        "Bajaj Finance": "BAJFINANCE.NS",
        "Bajaj Finserv": "BAJAJFINSV.NS",
        "HDFC Life": "HDFCLIFE.NS",
        "SBI Life": "SBILIFE.NS",
        "ICICI Lombard": "ICICIGI.NS",
        "ICICI Prudential": "ICICIPRULI.NS",
        "LIC": "LICI.NS",
        "Bajaj Holdings": "BAJAJHLDNG.NS",
        "Cholamandalam": "CHOLAFIN.NS",
        "SBI Cards": "SBICARD.NS",
        "Shriram Finance": "SHRIRAMFIN.NS",
        "HDFC AMC": "HDFCAMC.NS",
        "L&T Finance": "LTF.NS",
        "Manappuram Finance": "MANAPPURAM.NS",
        "Muthoot Finance": "MUTHOOTFIN.NS",
        "Nippon Life AMC": "NAM-INDIA.NS",
        "Sundaram Finance": "SUNDARMFIN.NS",
        "Can Fin Homes": "CANFINHOME.NS",
        "PFC": "PFC.NS",
        "REC": "RECLTD.NS",
        "Angel One": "ANGELONE.NS",
        "CreditAccess Grameen": "CREDITACC.NS",
        "Five Star Finance": "FIVESTAR.NS",
        "Go Digit": "GODIGIT.NS",
        "IIFL Finance": "IIFL.NS",
        "Motilal Oswal": "MOTILALOFS.NS",
        "PNB Housing Finance": "PNBHOUSING.NS",
        "Poonawalla Fincorp": "POONAWALLA.NS",
        "UTI AMC": "UTIAMC.NS",
        "HUDCO": "HUDCO.NS",
    },
    "Metals & Mining": {
        "Hindalco": "HINDALCO.NS",
        "JSW Steel": "JSWSTEEL.NS",
        "Tata Steel": "TATASTEEL.NS",
        "Jindal Steel": "JINDALSTEL.NS",
        "Vedanta": "VEDL.NS",
        "National Aluminium": "NATIONALUM.NS",
        "Sail": "SAIL.NS",
        "MOIL": "MOIL.NS",
        "NMDC Steel": "NMDCSTEEL.NS",
        "Shyam Metalics": "SHYAMMETL.NS",
        "Lloyds Metals": "LLOYDSMETA.NS",
        "Gravita India": "GRAVITA.NS",
        "Rajesh Exports": "RAJESHEXPO.NS",
    },
    "Infrastructure / Construction": {
        "L&T": "LT.NS",
        "Adani Ports": "ADANIPORTS.NS",
        "Grasim Industries": "GRASIM.NS",
        "GMR Airports": "GMRINFRA.NS",
        "NBCC": "NBCC.NS",
        "NCC": "NCC.NS",
        "KEC International": "KEC.NS",
        "Astral": "ASTRAL.NS",
        "Polycab India": "POLYCAB.NS",
        "KEI Industries": "KEI.NS",
        "Supreme Industries": "SUPREMEIND.NS",
        "Thermax": "THERMAX.NS",
        "Kajaria Ceramics": "KAJARIACER.NS",
        "Finolex Cables": "FINCABLES.NS",
    },
    "Cement": {
        "UltraTech Cement": "ULTRACEMCO.NS",
        "Ambuja Cements": "AMBUJACEM.NS",
        "Shree Cement": "SHREECEM.NS",
        "ACC": "ACC.NS",
        "JK Cement": "JKCEMENT.NS",
    },
    "Chemicals": {
        "UPL": "UPL.NS",
        "SRF": "SRF.NS",
        "Aarti Industries": "AARTIIND.NS",
        "Clean Science": "CLEAN.NS",
        "Deepak Nitrite": "DEEPAKNTR.NS",
        "Gujarat Fluorochemicals": "FLUOROCHEM.NS",
        "PI Industries": "PIIND.NS",
        "Tata Chemicals": "TATACHEM.NS",
        "Rashtriya Chemicals": "RCF.NS",
        "Deepak Fertilisers": "DEEPAKFERT.NS",
        "Jubilant Ingrevia": "JUBLINGREA.NS",
        "Sumitomo Chemical": "SUMICHEM.NS",
        "PCBL": "PCBL.NS",
        "Himadri Speciality": "HSCL.NS",
    },
    "Real Estate": {
        "DLF": "DLF.NS",
        "Godrej Properties": "GODREJPROP.NS",
        "Oberoi Realty": "OBEROIRLTY.NS",
        "Prestige Estates": "PRESTIGE.NS",
        "Sobha": "SOBHA.NS",
    },
    "Telecom": {
        "Bharti Airtel": "BHARTIARTL.NS",
        "Indus Towers": "INDUSTOWER.NS",
        "HFCL": "HFCL.NS",
        "ITI Limited": "ITI.NS",
        "Tejas Networks": "TEJASNET.NS",
    },
    "Defence": {
        "HAL": "HAL.NS",
        "BEL": "BEL.NS",
        "Data Patterns": "DATAPATTNS.NS",
        "Cochin Shipyard": "COCHINSHIP.NS",
        "GRSE": "GRSE.NS",
        "Mazagon Dock": "MAZDOCK.NS",
        "Solar Industries": "SOLARINDS.NS",
    },
    "Railways / Transport": {
        "IRCTC": "IRCTC.NS",
        "InterGlobe Aviation": "INDIGO.NS",
        "Delhivery": "DELHIVERY.NS",
        "RITES": "RITES.NS",
        "RVNL": "RVNL.NS",
        "IRCON International": "IRCON.NS",
        "IRFC": "IRFC.NS",
        "Jupiter Wagons": "JWL.NS",
        "Titagarh Rail": "TITAGARH.NS",
        "EaseMyTrip": "EASEMYTRIP.NS",
        "BLS International": "BLS.NS",
    },
    "Capital Goods / Industrial": {
        "Siemens": "SIEMENS.NS",
        "ABB India": "ABB.NS",
        "CG Power": "CGPOWER.NS",
        "Cummins India": "CUMMINSIND.NS",
        "Honeywell Automation": "HONAUT.NS",
        "Voltas": "VOLTAS.NS",
        "Carborundum Universal": "CARBORUNIV.NS",
        "Timken India": "TIMKEN.NS",
        "Triveni Turbine": "TRITURBINE.NS",
        "Elgi Equipments": "ELGIEQUIP.NS",
        "Kirloskar Brothers": "KIRLOSBROS.NS",
        "HBL Power": "HBLPOWER.NS",
        "Engineers India": "ENGINERSIN.NS",
    },
    "Media / Entertainment": {
        "Zee Entertainment": "ZEEL.NS",
        "PVR Inox": "PVRINOX.NS",
        "Vaibhav Global": "VAIBHAVGBL.NS",
    },
    "Hospitality / Travel": {
        "Indian Hotels": "INDHOTEL.NS",
        "Chalet Hotels": "CHALET.NS",
        "Lemon Tree Hotels": "LEMONTREE.NS",
    },
    "Textiles": {
        "Trident": "TRIDENT.NS",
        "Vardhman Textiles": "VTL.NS",
        "Welspun Living": "WELSPUNLIV.NS",
    },
    "Platform / Digital": {
        "Zomato": "ZOMATO.NS",
        "Paytm": "PAYTM.NS",
        "IndiaMart": "INDIAMART.NS",
        "BSE": "BSE.NS",
        "Central Depository": "CDSL.NS",
        "Multi Commodity Exch": "MCX.NS",
        "Indian Energy Exchange": "IEX.NS",
        "Computer Age Mgmt": "CAMS.NS",
        "Quess Corp": "QUESS.NS",
    },
}

US_STOCK_SECTORS = {
    "Technology": {
        "Apple": "AAPL", "Microsoft": "MSFT", "NVIDIA": "NVDA",
        "Alphabet": "GOOGL", "Meta Platforms": "META", "Broadcom": "AVGO",
        "Adobe": "ADBE", "Salesforce": "CRM", "AMD": "AMD",
        "Intel": "INTC", "Cisco": "CSCO", "Oracle": "ORCL",
        "IBM": "IBM", "Qualcomm": "QCOM", "Texas Instruments": "TXN",
    },
    "Healthcare": {
        "UnitedHealth": "UNH", "Johnson & Johnson": "JNJ", "Eli Lilly": "LLY",
        "Pfizer": "PFE", "AbbVie": "ABBV", "Merck": "MRK",
        "Thermo Fisher": "TMO", "Abbott Laboratories": "ABT",
        "Amgen": "AMGN", "Gilead Sciences": "GILD", "Moderna": "MRNA",
        "Intuitive Surgical": "ISRG", "Danaher": "DHR",
    },
    "Financials": {
        "JPMorgan Chase": "JPM", "Bank of America": "BAC", "Wells Fargo": "WFC",
        "Goldman Sachs": "GS", "Morgan Stanley": "MS", "Citigroup": "C",
        "BlackRock": "BLK", "Charles Schwab": "SCHW", "American Express": "AXP",
        "Visa": "V", "Mastercard": "MA", "PayPal": "PYPL",
        "S&P Global": "SPGI", "CME Group": "CME",
    },
    "Consumer Discretionary": {
        "Amazon": "AMZN", "Tesla": "TSLA", "Home Depot": "HD",
        "McDonald's": "MCD", "Nike": "NKE", "Starbucks": "SBUX",
        "Booking Holdings": "BKNG", "Lowe's": "LOW", "TJX Companies": "TJX",
        "Chipotle": "CMG", "Ross Stores": "ROST", "Marriott": "MAR",
    },
    "Consumer Staples": {
        "Procter & Gamble": "PG", "Coca-Cola": "KO", "PepsiCo": "PEP",
        "Costco": "COST", "Walmart": "WMT", "Colgate-Palmolive": "CL",
        "Mondelez": "MDLZ", "Philip Morris": "PM", "Altria": "MO",
        "Estee Lauder": "EL", "General Mills": "GIS", "Kraft Heinz": "KHC",
    },
    "Energy": {
        "ExxonMobil": "XOM", "Chevron": "CVX", "ConocoPhillips": "COP",
        "Schlumberger": "SLB", "EOG Resources": "EOG", "Pioneer Natural": "PXD",
        "Marathon Petroleum": "MPC", "Valero Energy": "VLO",
        "Phillips 66": "PSX", "Devon Energy": "DVN", "Hess": "HES",
    },
    "Industrials": {
        "Boeing": "BA", "Caterpillar": "CAT", "Honeywell": "HON",
        "Union Pacific": "UNP", "Lockheed Martin": "LMT", "Raytheon": "RTX",
        "3M": "MMM", "General Electric": "GE", "Deere & Company": "DE",
        "FedEx": "FDX", "UPS": "UPS", "Northrop Grumman": "NOC",
    },
    "Communication Services": {
        "Alphabet (GOOG)": "GOOG", "Meta Platforms (META)": "META",
        "Netflix": "NFLX", "Walt Disney": "DIS", "Comcast": "CMCSA",
        "T-Mobile": "TMUS", "AT&T": "T", "Verizon": "VZ",
        "Activision Blizzard": "ATVI", "Electronic Arts": "EA",
        "Warner Bros Discovery": "WBD",
    },
    "Utilities": {
        "NextEra Energy": "NEE", "Duke Energy": "DUK", "Southern Company": "SO",
        "Dominion Energy": "D", "Exelon": "EXC", "American Electric Power": "AEP",
        "Sempra": "SRE", "Xcel Energy": "XEL", "WEC Energy": "WEC",
        "Eversource Energy": "ES",
    },
    "Real Estate": {
        "Prologis": "PLD", "American Tower": "AMT", "Crown Castle": "CCI",
        "Equinix": "EQIX", "Public Storage": "PSA", "Simon Property": "SPG",
        "Realty Income": "O", "Digital Realty": "DLR", "Welltower": "WELL",
        "VICI Properties": "VICI",
    },
}

# =============================================================================
# Section 3: Helper Functions
# =============================================================================

def resolve_ticker(query, market="NSE"):
    query_upper = query.upper().strip()
    
    # 1. Exact or partial match in pre-defined dictionaries
    if market == "NSE":
        for stocks in [NIFTY_50, NIFTY_NEXT_50, MIDCAP_STOCKS, SMALLCAP_STOCKS, MICROCAP_STOCKS]:
            for name, ticker in stocks.items():
                if query_upper == name.upper() or query_upper == ticker.upper().replace(".NS", ""):
                    return ticker, name
        for sector, stocks in STOCK_SECTORS.items():
            for name, ticker in stocks.items():
                if query_upper == name.upper() or query_upper == ticker.upper().replace(".NS", ""):
                    return ticker, name
        for stocks in [NIFTY_50, NIFTY_NEXT_50, MIDCAP_STOCKS, SMALLCAP_STOCKS, MICROCAP_STOCKS]:
            for name, ticker in stocks.items():
                if query_upper in name.upper():
                    return ticker, name
    else:
        for sector, stocks in US_STOCK_SECTORS.items():
            for name, ticker in stocks.items():
                if query_upper == name.upper() or query_upper == ticker.upper():
                    return ticker, name
        for sector, stocks in US_STOCK_SECTORS.items():
            for name, ticker in stocks.items():
                if query_upper in name.upper():
                    return ticker, name

    # 2. Yahoo Finance Search API as fallback
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={quote_plus(query)}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
        quotes = res.get('quotes', [])
        if quotes:
            for q in quotes:
                sym = q['symbol']
                if market == "NSE" and sym.endswith(".NS"):
                    return sym, q.get('shortname', sym)
                elif market == "US" and not "." in sym:
                    return sym, q.get('shortname', sym)
    except Exception:
        pass

    # 3. Ultimate Fallback (assume what user entered is a valid symbol)
    if market == "NSE":
        # Block known US stocks
        for sector, stocks in US_STOCK_SECTORS.items():
            if query_upper in stocks.values() or query_upper in [n.upper() for n in stocks.keys()]:
                return None, None

        ticker = query_upper if query_upper.endswith(".NS") else f"{query_upper}.NS"
        return ticker, query_upper.replace(".NS", "")
    else:
        # Block known India stocks
        for stocks in [NIFTY_50, NIFTY_NEXT_50, MIDCAP_STOCKS, SMALLCAP_STOCKS, MICROCAP_STOCKS]:
            if query_upper in [t.replace(".NS", "") for t in stocks.values()] or query_upper in [n.upper() for n in stocks.keys()]:
                return None, None
                
        return query_upper, query_upper

def get_stock_sector(info):
    """Extract sector from yfinance info dict."""
    return info.get("sector") or info.get("industry") or "General"


# =============================================================================
# Section 4: Data Fetching (cached)
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_data(ticker):
    """Fetch price history, company info, and quarterly financials."""
    try:
        stock = _yf().Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty:
            return {"status": "error", "message": "No price data available"}

        info = stock.info or {}

        quarterly_income = None
        try:
            qi = stock.quarterly_income_stmt
            if qi is not None and not qi.empty:
                quarterly_income = qi
        except Exception:
            pass

        quarterly_balance = None
        try:
            qb = stock.quarterly_balance_sheet
            if qb is not None and not qb.empty:
                quarterly_balance = qb
        except Exception:
            pass

        annual_income = None
        try:
            ai = stock.income_stmt
            if ai is not None and not ai.empty:
                annual_income = ai
        except Exception:
            pass

        annual_balance = None
        try:
            ab = stock.balance_sheet
            if ab is not None and not ab.empty:
                annual_balance = ab
        except Exception:
            pass

        cashflow = None
        try:
            cf = stock.cashflow
            if cf is not None and not cf.empty:
                cashflow = cf
        except Exception:
            pass

        return {
            "status": "ok",
            "history": hist,
            "info": info,
            "quarterly_income": quarterly_income,
            "quarterly_balance": quarterly_balance,
            "annual_income": annual_income,
            "annual_balance": annual_balance,
            "cashflow": cashflow,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _parse_nse_index_csv(text):
    """Parse NSE index constituent CSV into {name: ticker} dict."""
    import pandas as pd
    df = pd.read_csv(io.StringIO(text))
    stocks = {}
    for _, row in df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        name = str(row.get("Company Name", "")).strip()
        if symbol and name and symbol != "nan" and name != "nan":
            stocks[name] = f"{symbol}.NS"
    return stocks


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nse_index_constituents(index_key):
    """Fetch live NSE index constituents — tries NSE archives first, then niftyindices.com."""
    # NSE archives (same domain as EQUITY_L.csv which works on Railway)
    archive_urls = {
        "midcap150":   "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
        "smallcap250": "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
        "microcap250": "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250list.csv",
        "nifty500":    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "niftynext50": "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
    }
    # niftyindices.com as secondary fallback
    niftyindices_urls = {
        "midcap150":   "https://www.niftyindices.com/IndexConstituents/ind_niftymidcap150list.csv",
        "smallcap250": "https://www.niftyindices.com/IndexConstituents/ind_niftysmallcap250list.csv",
        "microcap250": "https://www.niftyindices.com/IndexConstituents/ind_niftymicrocap250list.csv",
        "nifty500":    "https://www.niftyindices.com/IndexConstituents/ind_nifty500list.csv",
        "niftynext50": "https://www.niftyindices.com/IndexConstituents/ind_niftynext50list.csv",
    }

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for url_map, extra_headers in [
        (archive_urls, {}),
        (niftyindices_urls, {"Referer": "https://www.niftyindices.com/"}),
    ]:
        url = url_map.get(index_key)
        if not url:
            continue
        try:
            resp = requests.get(url, headers={**headers, **extra_headers}, timeout=12)
            resp.raise_for_status()
            result = _parse_nse_index_csv(resp.text)
            if result:
                return result
        except Exception:
            continue

    return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_all_nse_stocks():
    """Fetch complete list of NSE-listed equities from official NSE CSV."""
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        stocks = {}
        for _, row in df.iterrows():
            symbol = str(row.get("SYMBOL", "")).strip()
            name = str(row.get("NAME OF COMPANY", "")).strip()
            if symbol and name and row.get(" SERIES") and "EQ" in str(row.get(" SERIES")):
                stocks[name] = f"{symbol}.NS"
        return stocks
    except Exception:
        return {}


def _fetch_wiki_html(url):
    """Fetch Wikipedia page HTML with a proper User-Agent header."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockPredictor/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.text


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_stocks():
    """Fetch S&P 500 constituents from Wikipedia."""
    try:
        html = _fetch_wiki_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        dfs = pd.read_html(io.StringIO(html))
        df = dfs[0]
        return {row["Security"]: row["Symbol"] for _, row in df.iterrows()}
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nasdaq100_stocks():
    """Fetch NASDAQ 100 constituents from Wikipedia."""
    try:
        html = _fetch_wiki_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        dfs = pd.read_html(io.StringIO(html))
        for df in dfs:
            if "Ticker" in df.columns and "Company" in df.columns:
                return {row["Company"]: row["Ticker"] for _, row in df.iterrows()}
        return {}
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_dow30_stocks():
    """Fetch Dow 30 constituents from Wikipedia."""
    try:
        html = _fetch_wiki_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
        dfs = pd.read_html(io.StringIO(html))
        for df in dfs:
            if "Symbol" in df.columns and "Company" in df.columns:
                return {row["Company"]: row["Symbol"] for _, row in df.iterrows()}
        return {}
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_exchange_stocks(exchange):
    """Fetch all stocks for a given exchange (nyse/nasdaq) from NASDAQ API."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockPredictor/1.0)"}
    url = f"https://api.nasdaq.com/api/screener/stocks?tableType=traded&exchange={exchange}&limit=10000"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data", {}).get("table", {}).get("rows", [])
        stocks = {}
        for row in rows:
            symbol = row.get("symbol", "").strip()
            name = row.get("name", "").strip()
            if symbol and name:
                stocks[name] = symbol
        return stocks
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nyse_stocks():
    """Fetch all NYSE-listed stocks."""
    return _fetch_exchange_stocks("nyse")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nasdaq_stocks():
    """Fetch all NASDAQ-listed stocks."""
    return _fetch_exchange_stocks("nasdaq")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_headlines(stock_name, sector, market="India"):
    """Fetch news from Google News RSS for stock, sector, and market."""
    results = {"stock": [], "sector": [], "market": []}

    if market == "US":
        queries = {
            "stock": f"{stock_name} stock NYSE NASDAQ",
            "sector": f"US {sector} sector stock market",
            "market": "US stock market S&P 500 economy Federal Reserve",
        }
        hl, gl, ceid = "en-US", "US", "US:en"
    else:
        queries = {
            "stock": f"{stock_name} NSE stock",
            "sector": f"India {sector} sector stock market",
            "market": "Indian stock market Nifty economy policy",
        }
        hl, gl, ceid = "en-IN", "IN", "IN:en"

    for category, query in queries.items():
        try:
            url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"
            feed = _feedparser().parse(url)
            headlines = []
            for entry in feed.entries[:10]:
                headlines.append({
                    "title": entry.get("title", ""),
                    "published": entry.get("published", ""),
                    "source": entry.get("source", {}).get("title", "Unknown"),
                })
            results[category] = headlines
        except Exception:
            results[category] = []

    return results


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_global_macro_news(market="India"):
    """Fetch global macro-economic news: Fed, geopolitics, commodities, currencies."""
    if market == "India":
        queries = [
            "US Federal Reserve interest rate economy 2025",
            "India economy RBI GDP growth inflation",
            "global stock market crash rally recession",
            "crude oil gold silver commodity prices",
            "US China trade war tariffs geopolitics",
            "dollar rupee currency exchange rate",
            "FII DII foreign institutional investor India",
            "Middle East Russia Ukraine war global impact",
        ]
        hl, gl, ceid = "en-IN", "IN", "IN:en"
    else:
        queries = [
            "Federal Reserve interest rate inflation economy",
            "US GDP growth recession 2025",
            "global stock market outlook S&P",
            "crude oil OPEC commodity prices",
            "China economy trade war geopolitics",
            "dollar currency forex exchange",
            "earnings season tech AI sector",
            "geopolitical risk Middle East Europe",
        ]
        hl, gl, ceid = "en-US", "US", "US:en"

    all_headlines = []
    for query in queries:
        try:
            url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"
            feed = _feedparser().parse(url)
            for entry in feed.entries[:4]:
                all_headlines.append({
                    "title": entry.get("title", ""),
                    "published": entry.get("published", ""),
                    "source": entry.get("source", {}).get("title", "Unknown"),
                    "category": "macro",
                })
        except Exception:
            continue
    return all_headlines


@st.cache_data(ttl=600, show_spinner=False)
def fetch_oi_data(symbol_base):
    """Fetch Put-Call Ratio from NSE option chain (India F&O stocks only)."""
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.nseindia.com/option-chain",
            "Connection": "keep-alive",
        }
        try:
            session.get("https://www.nseindia.com", headers=headers, timeout=3)
        except Exception:
            pass  # Proceed without cookies; NSE may still respond
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol_base.upper()}"
        resp = session.get(url, headers=headers, timeout=4)
        if resp.status_code != 200:
            return {"status": "error"}
        data = resp.json()
        records = data.get("records", {}).get("data", [])
        total_call_oi = sum(r.get("CE", {}).get("openInterest", 0) for r in records if r.get("CE"))
        total_put_oi = sum(r.get("PE", {}).get("openInterest", 0) for r in records if r.get("PE"))
        total_call_chg = sum(r.get("CE", {}).get("changeinOpenInterest", 0) for r in records if r.get("CE"))
        total_put_chg = sum(r.get("PE", {}).get("changeinOpenInterest", 0) for r in records if r.get("PE"))
        if total_call_oi <= 0:
            return {"status": "error"}
        pcr = total_put_oi / total_call_oi
        pcr_chg = (total_put_chg / total_call_chg) if total_call_chg and total_call_chg != 0 else None
        # PCR > 1.2 → bullish (heavy put buying = hedging, smart money long)
        # PCR < 0.8 → bearish
        if pcr > 1.3:
            oi_signal = min((pcr - 1.0) * 0.8, 1.0)
        elif pcr > 1.0:
            oi_signal = (pcr - 1.0) * 0.4
        elif pcr < 0.7:
            oi_signal = max((pcr - 1.0) * 0.8, -1.0)
        elif pcr < 1.0:
            oi_signal = (pcr - 1.0) * 0.4
        else:
            oi_signal = 0.0
        return {
            "status": "ok",
            "pcr": round(pcr, 3),
            "pcr_change": round(pcr_chg, 3) if pcr_chg else None,
            "call_oi": total_call_oi,
            "put_oi": total_put_oi,
            "oi_signal": float(np.clip(oi_signal, -1, 1)),
        }
    except Exception:
        return {"status": "error"}


# =============================================================================
# Section 5: Technical Analysis
# =============================================================================

def calculate_technical_indicators(price_df):
    """Calculate enhanced technical signals: SMA, RSI, MACD, Bollinger Bands, OBV, ATR, momentum."""
    if price_df is None or len(price_df) < SMA_LONG:
        return {"status": "insufficient_data"}

    close = price_df["Close"]
    high = price_df["High"]
    low = price_df["Low"]
    volume = price_df.get("Volume", pd.Series(dtype=float))

    # --- SMA Crossover ---
    sma_short = close.rolling(window=SMA_SHORT).mean()
    sma_long = close.rolling(window=SMA_LONG).mean()
    latest_short = float(sma_short.iloc[-1])
    latest_long = float(sma_long.iloc[-1])
    sma_signal = float(np.clip((latest_short - latest_long) / latest_long * 10, -1, 1)) if latest_long else 0.0

    # --- RSI ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=RSI_PERIOD).mean()
    latest_loss = float(loss.iloc[-1])
    rsi = 100.0 if latest_loss == 0 else 100.0 - (100.0 / (1.0 + float(gain.iloc[-1]) / latest_loss))
    if rsi >= 70:
        rsi_signal = -((rsi - 70) / 30)
    elif rsi <= 30:
        rsi_signal = (30 - rsi) / 30
    else:
        rsi_signal = (rsi - 50) / 40
    rsi_signal = float(np.clip(rsi_signal, -1, 1))

    # --- MACD ---
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    macd_hist = macd_line - signal_line
    latest_macd = float(macd_hist.iloc[-1])
    prev_macd = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else latest_macd
    # Normalize by price to make scale-independent
    price_scale = float(close.iloc[-1])
    if price_scale and price_scale != 0:
        macd_norm = latest_macd / price_scale * 100
        macd_trend = (latest_macd - prev_macd) / price_scale * 200
        macd_signal_val = float(np.clip(macd_norm * 3 + macd_trend, -1, 1))
    else:
        macd_signal_val = 0.0
    macd_crossover = "Bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "Bearish"

    # --- Bollinger Bands ---
    bb_mid = close.rolling(window=BB_PERIOD).mean()
    bb_std_val = close.rolling(window=BB_PERIOD).std()
    bb_upper = bb_mid + BB_STD * bb_std_val
    bb_lower = bb_mid - BB_STD * bb_std_val
    latest_price = float(close.iloc[-1])
    bb_u = float(bb_upper.iloc[-1])
    bb_l = float(bb_lower.iloc[-1])
    bb_m = float(bb_mid.iloc[-1])
    bb_width = bb_u - bb_l
    if bb_width > 0:
        bb_pct = (latest_price - bb_l) / bb_width  # 0=at lower, 1=at upper
        if bb_pct <= 0.1:
            bb_signal_val = 1.0   # At/below lower band → oversold
        elif bb_pct >= 0.9:
            bb_signal_val = -1.0  # At/above upper band → overbought
        elif bb_pct <= 0.3:
            bb_signal_val = 0.5
        elif bb_pct >= 0.7:
            bb_signal_val = -0.5
        else:
            bb_signal_val = (0.5 - bb_pct) * 2.0  # slight lean based on position
    else:
        bb_pct = 0.5
        bb_signal_val = 0.0
    bb_signal_val = float(np.clip(bb_signal_val, -1, 1))

    # --- OBV (On-Balance Volume) trend ---
    obv_signal_val = 0.0
    obv_trend_label = "N/A"
    if volume is not None and not volume.empty and len(volume) >= VOLUME_MA_PERIOD:
        close_diff = close.diff()
        obv = (np.where(close_diff > 0, volume, np.where(close_diff < 0, -volume, 0))).cumsum()
        obv_series = pd.Series(obv, index=close.index)
        obv_ma = obv_series.rolling(window=VOLUME_MA_PERIOD).mean()
        latest_obv = float(obv_series.iloc[-1])
        latest_obv_ma = float(obv_ma.iloc[-1])
        if latest_obv_ma != 0:
            obv_diff = (latest_obv - latest_obv_ma) / abs(latest_obv_ma)
            obv_signal_val = float(np.clip(obv_diff * 3, -1, 1))
            obv_trend_label = "Accumulation" if obv_signal_val > 0.1 else "Distribution" if obv_signal_val < -0.1 else "Neutral"

        # Volume surge detection (current volume vs 20-day avg)
        vol_ma = volume.rolling(window=VOLUME_MA_PERIOD).mean()
        latest_vol = float(volume.iloc[-1])
        avg_vol = float(vol_ma.iloc[-1]) if not pd.isna(vol_ma.iloc[-1]) else latest_vol
        volume_ratio = round(latest_vol / avg_vol, 2) if avg_vol > 0 else 1.0
    else:
        volume_ratio = 1.0

    # --- ATR (Average True Range) ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(window=ATR_PERIOD).mean().iloc[-1])

    # --- Momentum Signal (5-day vs 20-day return) ---
    if len(close) >= 20:
        ret_5d = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if close.iloc[-5] != 0 else 0
        ret_20d = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if close.iloc[-20] != 0 else 0
        momentum_signal = float(np.clip((ret_5d * 0.6 + ret_20d * 0.4) * 5, -1, 1))
    else:
        momentum_signal = 0.0

    # --- 52-week position ---
    w52_high = float(high.max())
    w52_low = float(low.min())
    w52_range = w52_high - w52_low
    w52_pct = (latest_price - w52_low) / w52_range if w52_range > 0 else 0.5  # 0=at low, 1=at high

    # --- Composite technical score ---
    composite = (
        sma_signal * TECH_SMA_WEIGHT
        + rsi_signal * TECH_RSI_WEIGHT
        + macd_signal_val * TECH_MACD_WEIGHT
        + bb_signal_val * TECH_BB_WEIGHT
        + obv_signal_val * TECH_OBV_WEIGHT
        + momentum_signal * TECH_MOMENTUM_WEIGHT
    )

    return {
        "status": "ok",
        "score": float(np.clip(composite, -1, 1)),
        # SMA
        "sma_signal": sma_signal,
        "sma_short": latest_short,
        "sma_long": latest_long,
        # RSI
        "rsi_signal": rsi_signal,
        "rsi_value": rsi,
        # MACD
        "macd_signal": macd_signal_val,
        "macd_line": float(macd_line.iloc[-1]),
        "macd_signal_line": float(signal_line.iloc[-1]),
        "macd_hist": latest_macd,
        "macd_crossover": macd_crossover,
        # Bollinger Bands
        "bb_signal": bb_signal_val,
        "bb_upper": bb_u,
        "bb_lower": bb_l,
        "bb_mid": bb_m,
        "bb_pct": round(bb_pct * 100, 1),
        # OBV / Volume
        "obv_signal": obv_signal_val,
        "obv_trend": obv_trend_label,
        "volume_ratio": volume_ratio,
        # Momentum & Price
        "momentum_signal": momentum_signal,
        "current_price": latest_price,
        "atr": round(atr, 2),
        "w52_high": round(w52_high, 2),
        "w52_low": round(w52_low, 2),
        "w52_pct": round(w52_pct * 100, 1),
    }


def calculate_trade_levels(technical, prediction_action):
    """Calculate entry zone, target prices, and stop loss using support/resistance + ATR."""
    price = technical.get("current_price")
    atr = technical.get("atr", 0)
    w52_high = technical.get("w52_high")
    w52_low = technical.get("w52_low")
    bb_upper = technical.get("bb_upper")
    bb_lower = technical.get("bb_lower")
    sma_long = technical.get("sma_long")

    if not price or not atr:
        return None

    # Estimate support and resistance from available levels
    resistance_candidates = [x for x in [bb_upper, w52_high, sma_long * 1.03] if x and x > price]
    support_candidates = [x for x in [bb_lower, w52_low, sma_long * 0.97] if x and x < price]

    nearest_resistance = min(resistance_candidates) if resistance_candidates else price * 1.08
    nearest_support = max(support_candidates) if support_candidates else price * 0.93

    if prediction_action == "BULLISH":
        entry_low = round(price * 0.99, 2)
        entry_high = round(price * 1.005, 2)
        stop_loss = round(max(nearest_support - atr * 0.5, price * 0.92), 2)
        target1 = round(min(nearest_resistance, price * 1.07), 2)
        risk = price - stop_loss
        target2 = round(price + risk * 2.5, 2)
        target3 = round(price + risk * 4.0, 2)
    elif prediction_action == "BEARISH":
        entry_low = round(price * 0.995, 2)
        entry_high = round(price * 1.01, 2)
        stop_loss = round(min(nearest_resistance + atr * 0.5, price * 1.08), 2)
        target1 = round(max(nearest_support, price * 0.93), 2)
        risk = stop_loss - price
        target2 = round(price - risk * 2.5, 2)
        target3 = round(price - risk * 4.0, 2)
    else:
        return None

    risk_val = abs(price - stop_loss)
    reward1 = abs(target1 - price)
    rr1 = round(reward1 / risk_val, 2) if risk_val > 0 else 0
    rr2 = round(abs(target2 - price) / risk_val, 2) if risk_val > 0 else 0

    return {
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "target3": target3,
        "risk_reward1": rr1,
        "risk_reward2": rr2,
        "risk_per_share": round(risk_val, 2),
        "nearest_support": round(nearest_support, 2),
        "nearest_resistance": round(nearest_resistance, 2),
    }


# =============================================================================
# Section 6: Sentiment Analysis
# =============================================================================

def analyze_sentiment(news_dict):
    """Analyze sentiment of news headlines using VADER."""
    analyzer = _get_vader_analyzer()

    category_scores = {}
    total_headlines = 0

    for category in ["stock", "sector", "market"]:
        headlines = news_dict.get(category, [])
        if not headlines:
            category_scores[category] = 0.0
            continue

        scores = []
        for item in headlines:
            title = item.get("title", "")
            if title:
                vs = analyzer.polarity_scores(title)
                scores.append(vs["compound"])
                item["sentiment_score"] = vs["compound"]

        category_scores[category] = float(np.mean(scores)) if scores else 0.0
        total_headlines += len(scores)

    if total_headlines == 0:
        return {"status": "insufficient_data"}

    composite = (
        category_scores.get("stock", 0) * SENTIMENT_STOCK_WEIGHT
        + category_scores.get("sector", 0) * SENTIMENT_SECTOR_WEIGHT
        + category_scores.get("market", 0) * SENTIMENT_MARKET_WEIGHT
    )

    return {
        "status": "ok",
        "score": float(np.clip(composite, -1, 1)),
        "stock_sentiment": category_scores.get("stock", 0),
        "sector_sentiment": category_scores.get("sector", 0),
        "market_sentiment": category_scores.get("market", 0),
        "headline_count": total_headlines,
        "news_with_scores": news_dict,
    }


# =============================================================================
# Section 7: Fundamental Analysis
# =============================================================================

def analyze_fundamentals(quarterly_df, info, balance_sheet_df=None):
    """Analyze QoQ revenue growth, profit margin, profit growth, D/E, current ratio, and ROE."""
    signals = {}
    available_signals = 0
    raw_de_ratio = None
    raw_current_ratio = None
    raw_roe = None

    # QoQ Revenue Growth
    try:
        if quarterly_df is not None and not quarterly_df.empty:
            revenue_row = None
            for label in ["Total Revenue", "TotalRevenue", "Revenue"]:
                if label in quarterly_df.index:
                    revenue_row = quarterly_df.loc[label]
                    break

            if revenue_row is not None and len(revenue_row) >= 2:
                latest_rev = revenue_row.iloc[0]
                prev_rev = revenue_row.iloc[1]
                if prev_rev and prev_rev != 0:
                    rev_growth = (latest_rev - prev_rev) / abs(prev_rev)
                    signals["revenue_growth"] = float(np.clip(rev_growth * 5, -1, 1))
                    available_signals += 1
    except Exception:
        pass

    # Profit Margin
    try:
        margin = info.get("profitMargins")
        if margin is not None:
            if margin > 0.20:
                signals["profit_margin"] = min(margin * 2, 1.0)
            elif margin > 0.10:
                signals["profit_margin"] = margin * 3 - 0.3
            elif margin > 0:
                signals["profit_margin"] = margin * 2 - 0.2
            else:
                signals["profit_margin"] = max(margin * 2, -1.0)
            signals["profit_margin"] = float(np.clip(signals["profit_margin"], -1, 1))
            available_signals += 1
    except Exception:
        pass

    # QoQ Profit Growth
    try:
        if quarterly_df is not None and not quarterly_df.empty:
            profit_row = None
            for label in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
                if label in quarterly_df.index:
                    profit_row = quarterly_df.loc[label]
                    break

            if profit_row is not None and len(profit_row) >= 2:
                latest_profit = profit_row.iloc[0]
                prev_profit = profit_row.iloc[1]
                if prev_profit and prev_profit != 0:
                    profit_growth = (latest_profit - prev_profit) / abs(prev_profit)
                    signals["profit_growth"] = float(np.clip(profit_growth * 5, -1, 1))
                    available_signals += 1
    except Exception:
        pass

    # Debt-to-Equity Ratio (from balance sheet)
    try:
        if balance_sheet_df is not None and not balance_sheet_df.empty:
            total_debt = None
            for label in ["Total Debt", "TotalDebt", "Long Term Debt And Capital Lease Obligation"]:
                if label in balance_sheet_df.index:
                    total_debt = balance_sheet_df.loc[label].iloc[0]
                    break

            stockholders_equity = None
            for label in ["Stockholders Equity", "StockholdersEquity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
                if label in balance_sheet_df.index:
                    stockholders_equity = balance_sheet_df.loc[label].iloc[0]
                    break

            if total_debt is not None and stockholders_equity is not None:
                if stockholders_equity <= 0:
                    signals["debt_to_equity"] = -1.0
                    raw_de_ratio = None  # Negative equity
                else:
                    de_ratio = total_debt / stockholders_equity
                    raw_de_ratio = float(de_ratio)
                    if de_ratio < 0.5:
                        signals["debt_to_equity"] = 1.0
                    elif de_ratio < 1.0:
                        signals["debt_to_equity"] = 0.5
                    elif de_ratio < 2.0:
                        signals["debt_to_equity"] = -0.3
                    else:
                        signals["debt_to_equity"] = -1.0
                available_signals += 1
    except Exception:
        pass

    # Current Ratio (from balance sheet)
    try:
        if balance_sheet_df is not None and not balance_sheet_df.empty:
            current_assets = None
            for label in ["Current Assets", "CurrentAssets", "Total Current Assets"]:
                if label in balance_sheet_df.index:
                    current_assets = balance_sheet_df.loc[label].iloc[0]
                    break

            current_liabilities = None
            for label in ["Current Liabilities", "CurrentLiabilities", "Total Current Liabilities"]:
                if label in balance_sheet_df.index:
                    current_liabilities = balance_sheet_df.loc[label].iloc[0]
                    break

            if current_assets is not None and current_liabilities is not None and current_liabilities != 0:
                cr = current_assets / current_liabilities
                raw_current_ratio = float(cr)
                if cr >= 2.0:
                    signals["current_ratio"] = 1.0
                elif cr >= 1.5:
                    signals["current_ratio"] = 0.6
                elif cr >= 1.0:
                    signals["current_ratio"] = 0.2
                elif cr >= 0.5:
                    signals["current_ratio"] = -0.5
                else:
                    signals["current_ratio"] = -1.0
                available_signals += 1
    except Exception:
        pass

    # Return on Equity (from info)
    try:
        roe = info.get("returnOnEquity")
        if roe is not None:
            raw_roe = float(roe)
            if roe >= 0.25:
                signals["roe"] = 1.0
            elif roe >= 0.15:
                signals["roe"] = 0.7
            elif roe >= 0.10:
                signals["roe"] = 0.4
            elif roe >= 0:
                signals["roe"] = 0.1
            else:
                signals["roe"] = max(roe * 3, -1.0)
            signals["roe"] = float(np.clip(signals["roe"], -1, 1))
            available_signals += 1
    except Exception:
        pass

    # Return on Capital Assets (ROCA = Net Income / Total Assets)
    raw_roca = None
    try:
        roa = info.get("returnOnAssets")
        if roa is not None:
            raw_roca = float(roa)
            if roa >= 0.15:
                signals["roca"] = 1.0
            elif roa >= 0.10:
                signals["roca"] = 0.7
            elif roa >= 0.05:
                signals["roca"] = 0.3
            elif roa >= 0:
                signals["roca"] = 0.0
            else:
                signals["roca"] = max(roa * 5, -1.0)
            signals["roca"] = float(np.clip(signals["roca"], -1, 1))
            available_signals += 1
        elif balance_sheet_df is not None and not balance_sheet_df.empty and quarterly_df is not None:
            # Compute from financial statements
            ni_row = None
            for lbl in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
                if lbl in quarterly_df.index:
                    ni_row = quarterly_df.loc[lbl]
                    break
            ta_row = None
            for lbl in ["Total Assets", "TotalAssets"]:
                if lbl in balance_sheet_df.index:
                    ta_row = balance_sheet_df.loc[lbl]
                    break
            if ni_row is not None and ta_row is not None:
                ni_val = float(ni_row.iloc[0])
                ta_val = float(ta_row.iloc[0])
                if ta_val and ta_val != 0:
                    raw_roca = ni_val / ta_val
                    signals["roca"] = float(np.clip(raw_roca * 7, -1, 1))
                    available_signals += 1
    except Exception:
        pass

    # Return on Capital Employed (ROCE = EBIT / Capital Employed)
    raw_roce = None
    try:
        if balance_sheet_df is not None and not balance_sheet_df.empty and quarterly_df is not None:
            ebit_row = None
            for lbl in ["EBIT", "Operating Income", "OperatingIncome"]:
                if lbl in quarterly_df.index:
                    ebit_row = quarterly_df.loc[lbl]
                    break
            ta_row = None
            for lbl in ["Total Assets", "TotalAssets"]:
                if lbl in balance_sheet_df.index:
                    ta_row = balance_sheet_df.loc[lbl]
                    break
            cl_row = None
            for lbl in ["Current Liabilities", "CurrentLiabilities", "Total Current Liabilities"]:
                if lbl in balance_sheet_df.index:
                    cl_row = balance_sheet_df.loc[lbl]
                    break
            if ebit_row is not None and ta_row is not None and cl_row is not None:
                ce = float(ta_row.iloc[0]) - float(cl_row.iloc[0])
                if ce and ce != 0:
                    raw_roce = float(ebit_row.iloc[0]) / ce
                    if raw_roce >= 0.20:
                        signals["roce"] = 1.0
                    elif raw_roce >= 0.12:
                        signals["roce"] = 0.6
                    elif raw_roce >= 0.08:
                        signals["roce"] = 0.2
                    elif raw_roce >= 0:
                        signals["roce"] = -0.1
                    else:
                        signals["roce"] = max(raw_roce * 5, -1.0)
                    signals["roce"] = float(np.clip(signals["roce"], -1, 1))
                    available_signals += 1
    except Exception:
        pass

    if available_signals == 0:
        return {"status": "insufficient_data"}

    # Compute composite using fixed weights when we have all 8, else equal weight
    all_weights = {
        "revenue_growth": FUND_REVENUE_WEIGHT,
        "profit_margin": FUND_MARGIN_WEIGHT,
        "profit_growth": FUND_PROFIT_WEIGHT,
        "debt_to_equity": FUND_DEBT_EQUITY_WEIGHT,
        "current_ratio": FUND_CURRENT_RATIO_WEIGHT,
        "roe": FUND_ROE_WEIGHT,
        "roca": FUND_ROCA_WEIGHT,
        "roce": FUND_ROCE_WEIGHT,
    }
    total_w = sum(all_weights[k] for k in signals if k in all_weights)
    if total_w > 0:
        composite = sum(signals[k] * all_weights.get(k, 0) / total_w for k in signals if k in all_weights)
    else:
        vals = list(signals.values())
        composite = sum(vals) / len(vals)

    return {
        "status": "ok",
        "score": float(np.clip(composite, -1, 1)),
        "revenue_growth": signals.get("revenue_growth"),
        "profit_margin": signals.get("profit_margin"),
        "profit_growth": signals.get("profit_growth"),
        "debt_to_equity": signals.get("debt_to_equity"),
        "current_ratio": signals.get("current_ratio"),
        "roe": signals.get("roe"),
        "roca": signals.get("roca"),
        "roce": signals.get("roce"),
        "available_signals": available_signals,
        "raw_margin": info.get("profitMargins"),
        "raw_de_ratio": raw_de_ratio,
        "raw_current_ratio": raw_current_ratio,
        "raw_roe": raw_roe,
        "raw_roca": raw_roca,
        "raw_roce": raw_roce,
    }


# =============================================================================
# Section 7b: Key Financial Metrics (Display-Only)
# =============================================================================

def _safe_get_row(df, labels):
    """Return the first matching row from a DataFrame given a list of label candidates."""
    if df is None or df.empty:
        return None
    for label in labels:
        if label in df.index:
            return df.loc[label]
    return None


def compute_key_metrics(info, annual_income, annual_balance, cashflow):
    """Compute display-only key financial metrics.

    Returns a dict of metric values (or None where unavailable).
    These do NOT affect the signal output.
    """
    metrics = {
        # Existing
        "pe_ratio": None,
        "price_to_book": None,
        "debt_to_equity": None,
        "current_ratio": None,
        "roe_latest": None,
        "roe_3yr": None,
        "roe_5yr": None,
        "roce_latest": None,
        "roce_3yr": None,
        "roce_5yr": None,
        "piotroski_score": None,
        "piotroski_details": None,
        "market_cap": None,
        # Valuation
        "ps_ratio": None,
        "ev_ebitda": None,
        "ev_revenue": None,
        "forward_pe": None,
        "enterprise_value": None,
        # Profitability (raw fractions, e.g. 0.25 = 25%)
        "gross_margin": None,
        "operating_margin": None,
        "net_margin": None,
        "roa": None,
        "ebitda_margin": None,
        # Growth (raw fractions, e.g. 0.12 = 12%)
        "revenue_growth_yoy": None,
        "earnings_growth_yoy": None,
        # Financial Health
        "quick_ratio": None,
        "interest_coverage": None,
        # Dividend (raw fractions)
        "dividend_yield": None,
        "payout_ratio": None,
        # Per Share / Risk
        "eps": None,
        "beta": None,
    }

    if not info:
        info = {}

    # --- Direct from yfinance info ---
    metrics["pe_ratio"] = info.get("trailingPE")
    metrics["price_to_book"] = info.get("priceToBook")

    de = info.get("debtToEquity")
    if de is not None:
        metrics["debt_to_equity"] = de / 100.0

    metrics["current_ratio"] = info.get("currentRatio")
    metrics["market_cap"] = info.get("marketCap")

    # Valuation
    metrics["ps_ratio"] = info.get("priceToSalesTrailing12Months")
    metrics["ev_ebitda"] = info.get("enterpriseToEbitda")
    metrics["ev_revenue"] = info.get("enterpriseToRevenue")
    metrics["forward_pe"] = info.get("forwardPE")
    metrics["enterprise_value"] = info.get("enterpriseValue")

    # Profitability
    metrics["gross_margin"] = info.get("grossMargins")
    metrics["operating_margin"] = info.get("operatingMargins")
    metrics["net_margin"] = info.get("profitMargins")
    metrics["roa"] = info.get("returnOnAssets")
    metrics["ebitda_margin"] = info.get("ebitdaMargins")

    # Growth
    metrics["revenue_growth_yoy"] = info.get("revenueGrowth")
    metrics["earnings_growth_yoy"] = info.get("earningsGrowth")

    # Financial Health
    metrics["quick_ratio"] = info.get("quickRatio")

    # Dividend
    dy = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
    metrics["dividend_yield"] = dy
    metrics["payout_ratio"] = info.get("payoutRatio")

    # Per Share / Risk
    metrics["eps"] = info.get("trailingEps")
    metrics["beta"] = info.get("beta")

    # --- Interest Coverage (EBIT / Interest Expense) from annual income stmt ---
    try:
        if annual_income is not None and not annual_income.empty:
            ebit_row = _safe_get_row(annual_income, ["EBIT", "Operating Income", "OperatingIncome"])
            interest_row = _safe_get_row(
                annual_income,
                ["Interest Expense", "InterestExpense", "Interest Expense Non Operating"]
            )
            if ebit_row is not None and interest_row is not None:
                ebit_val = float(ebit_row.iloc[0])
                interest_val = float(interest_row.iloc[0])
                # Interest expense is often stored as a negative number
                interest_abs = abs(interest_val)
                if interest_abs > 0:
                    metrics["interest_coverage"] = round(ebit_val / interest_abs, 2)
    except Exception:
        pass

    # --- Multi-year ROE ---
    try:
        if annual_income is not None and annual_balance is not None:
            ni_row = _safe_get_row(annual_income, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
            eq_row = _safe_get_row(annual_balance, ["Stockholders Equity", "StockholdersEquity",
                                                     "Total Equity Gross Minority Interest", "Common Stock Equity"])
            if ni_row is not None and eq_row is not None:
                roe_values = []
                n_years = min(len(ni_row), len(eq_row), 5)
                for i in range(n_years):
                    ni_val = ni_row.iloc[i]
                    eq_val = eq_row.iloc[i]
                    if eq_val and eq_val != 0:
                        roe_values.append(float(ni_val) / float(eq_val))

                if roe_values:
                    metrics["roe_latest"] = roe_values[0]
                if len(roe_values) >= 2:
                    metrics["roe_3yr"] = sum(roe_values[:min(3, len(roe_values))]) / min(3, len(roe_values))
                if len(roe_values) >= 4:
                    metrics["roe_5yr"] = sum(roe_values[:min(5, len(roe_values))]) / min(5, len(roe_values))
    except Exception:
        pass

    # --- Multi-year ROCE ---
    try:
        if annual_income is not None and annual_balance is not None:
            ebit_row = _safe_get_row(annual_income, ["EBIT", "Operating Income", "OperatingIncome"])
            ta_row = _safe_get_row(annual_balance, ["Total Assets", "TotalAssets"])
            cl_row = _safe_get_row(annual_balance, ["Current Liabilities", "CurrentLiabilities",
                                                     "Total Current Liabilities"])

            if ebit_row is not None and ta_row is not None and cl_row is not None:
                roce_values = []
                n_years = min(len(ebit_row), len(ta_row), len(cl_row), 5)
                for i in range(n_years):
                    ebit_val = ebit_row.iloc[i]
                    ta_val = ta_row.iloc[i]
                    cl_val = cl_row.iloc[i]
                    capital_employed = float(ta_val) - float(cl_val)
                    if capital_employed and capital_employed != 0:
                        roce_values.append(float(ebit_val) / capital_employed)

                if roce_values:
                    metrics["roce_latest"] = roce_values[0]
                if len(roce_values) >= 2:
                    metrics["roce_3yr"] = sum(roce_values[:min(3, len(roce_values))]) / min(3, len(roce_values))
                if len(roce_values) >= 4:
                    metrics["roce_5yr"] = sum(roce_values[:min(5, len(roce_values))]) / min(5, len(roce_values))
    except Exception:
        pass

    # --- Piotroski F-Score ---
    try:
        if annual_income is not None and annual_balance is not None and cashflow is not None:
            details = []
            score = 0

            ni_row = _safe_get_row(annual_income, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
            ta_row = _safe_get_row(annual_balance, ["Total Assets", "TotalAssets"])
            cfo_row = _safe_get_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities",
                                                "Cash Flow From Continuing Operating Activities"])
            ltd_row = _safe_get_row(annual_balance, ["Long Term Debt", "LongTermDebt",
                                                      "Long Term Debt And Capital Lease Obligation"])
            ca_row = _safe_get_row(annual_balance, ["Current Assets", "CurrentAssets", "Total Current Assets"])
            cl_row = _safe_get_row(annual_balance, ["Current Liabilities", "CurrentLiabilities",
                                                     "Total Current Liabilities"])
            shares_row = _safe_get_row(annual_balance, ["Ordinary Shares Number", "Share Issued",
                                                         "Common Stock"])
            gp_row = _safe_get_row(annual_income, ["Gross Profit", "GrossProfit"])
            rev_row = _safe_get_row(annual_income, ["Total Revenue", "TotalRevenue", "Revenue"])

            has_two_years = all(
                r is not None and len(r) >= 2
                for r in [ni_row, ta_row]
            )

            if has_two_years:
                # 1. Net Income > 0
                ni_positive = float(ni_row.iloc[0]) > 0
                details.append(("Net Income > 0", ni_positive))
                score += int(ni_positive)

                # 2. ROA > 0
                roa_positive = False
                if ta_row is not None and float(ta_row.iloc[0]) != 0:
                    roa_positive = (float(ni_row.iloc[0]) / float(ta_row.iloc[0])) > 0
                details.append(("ROA > 0", roa_positive))
                score += int(roa_positive)

                # 3. Operating Cash Flow > 0
                cfo_positive = False
                if cfo_row is not None and len(cfo_row) >= 1:
                    cfo_positive = float(cfo_row.iloc[0]) > 0
                details.append(("Operating Cash Flow > 0", cfo_positive))
                score += int(cfo_positive)

                # 4. CFO > Net Income (accruals)
                cfo_gt_ni = False
                if cfo_row is not None and len(cfo_row) >= 1:
                    cfo_gt_ni = float(cfo_row.iloc[0]) > float(ni_row.iloc[0])
                details.append(("Cash Flow > Net Income", cfo_gt_ni))
                score += int(cfo_gt_ni)

                # 5. Long-term debt decreased or same
                ltd_decreased = False
                if ltd_row is not None and len(ltd_row) >= 2:
                    ltd_decreased = float(ltd_row.iloc[0]) <= float(ltd_row.iloc[1])
                else:
                    ltd_decreased = True  # No debt info = assume no debt
                details.append(("Long-term Debt Decreased", ltd_decreased))
                score += int(ltd_decreased)

                # 6. Current ratio increased
                cr_increased = False
                if (ca_row is not None and cl_row is not None
                        and len(ca_row) >= 2 and len(cl_row) >= 2):
                    cl_curr = float(cl_row.iloc[0])
                    cl_prev = float(cl_row.iloc[1])
                    if cl_curr != 0 and cl_prev != 0:
                        cr_curr = float(ca_row.iloc[0]) / cl_curr
                        cr_prev = float(ca_row.iloc[1]) / cl_prev
                        cr_increased = cr_curr >= cr_prev
                details.append(("Current Ratio Increased", cr_increased))
                score += int(cr_increased)

                # 7. No new shares issued
                no_dilution = False
                if shares_row is not None and len(shares_row) >= 2:
                    no_dilution = float(shares_row.iloc[0]) <= float(shares_row.iloc[1])
                else:
                    no_dilution = True
                details.append(("No New Shares Issued", no_dilution))
                score += int(no_dilution)

                # 8. Gross margin increased
                gm_increased = False
                if (gp_row is not None and rev_row is not None
                        and len(gp_row) >= 2 and len(rev_row) >= 2):
                    rev_curr = float(rev_row.iloc[0])
                    rev_prev = float(rev_row.iloc[1])
                    if rev_curr != 0 and rev_prev != 0:
                        gm_curr = float(gp_row.iloc[0]) / rev_curr
                        gm_prev = float(gp_row.iloc[1]) / rev_prev
                        gm_increased = gm_curr >= gm_prev
                details.append(("Gross Margin Increased", gm_increased))
                score += int(gm_increased)

                # 9. Asset turnover increased
                at_increased = False
                if (rev_row is not None and ta_row is not None
                        and len(rev_row) >= 2 and len(ta_row) >= 2):
                    ta_curr = float(ta_row.iloc[0])
                    ta_prev = float(ta_row.iloc[1])
                    if ta_curr != 0 and ta_prev != 0:
                        at_curr = float(rev_row.iloc[0]) / ta_curr
                        at_prev = float(rev_row.iloc[1]) / ta_prev
                        at_increased = at_curr >= at_prev
                details.append(("Asset Turnover Increased", at_increased))
                score += int(at_increased)

                metrics["piotroski_score"] = score
                metrics["piotroski_details"] = details
    except Exception:
        pass

    return metrics



# =============================================================================
# Section 7b: Screen Builder - Saved Screens and Filter Logic
# =============================================================================

SAVED_SCREENS_FILE = "saved_screens.json"

SCREENER_FIELDS = {
    # Signal
    "Action":            {"type": "categorical", "values": ["BULLISH", "NEUTRAL", "BEARISH"]},
    "Score":             {"type": "numeric", "hint": "-1.0 to 1.0"},
    "Confidence":        {"type": "numeric", "hint": "0 to 100"},
    "Tech Score":        {"type": "numeric", "hint": "-1.0 to 1.0"},
    "Sentiment Score":   {"type": "numeric", "hint": "-1.0 to 1.0"},
    "Fundamental Score": {"type": "numeric", "hint": "-1.0 to 1.0"},
    # Price / Size
    "CMP":               {"type": "numeric", "hint": "Current Market Price"},
    "Market Cap":        {"type": "numeric", "hint": "in absolute value"},
    "52W High":          {"type": "numeric", "hint": "e.g. > 500"},
    "52W Low":           {"type": "numeric", "hint": "e.g. < 200"},
    "% from 52W High":   {"type": "numeric", "hint": "e.g. > -10 (negative means below high)"},
    # Valuation
    "P/E":               {"type": "numeric", "hint": "e.g. < 25"},
    "Fwd P/E":           {"type": "numeric", "hint": "e.g. < 20"},
    "P/B":               {"type": "numeric", "hint": "e.g. < 3"},
    "P/S":               {"type": "numeric", "hint": "e.g. < 5"},
    "EV/EBITDA":         {"type": "numeric", "hint": "e.g. < 15"},
    "EV/Revenue":        {"type": "numeric", "hint": "e.g. < 5"},
    # Profitability (values in %)
    "ROE":               {"type": "numeric", "hint": "e.g. > 15  (%)"},
    "ROCE":              {"type": "numeric", "hint": "e.g. > 12  (%)"},
    "ROA":               {"type": "numeric", "hint": "e.g. > 8   (%)"},
    "Gross Margin":      {"type": "numeric", "hint": "e.g. > 30  (%)"},
    "Operating Margin":  {"type": "numeric", "hint": "e.g. > 15  (%)"},
    "Net Margin":        {"type": "numeric", "hint": "e.g. > 10  (%)"},
    "EBITDA Margin":     {"type": "numeric", "hint": "e.g. > 20  (%)"},
    # Growth (values in %)
    "Rev Growth YoY":    {"type": "numeric", "hint": "e.g. > 10  (%)"},
    "EPS Growth YoY":    {"type": "numeric", "hint": "e.g. > 10  (%)"},
    "EPS":               {"type": "numeric", "hint": "e.g. > 10"},
    # Financial Health
    "D/E":               {"type": "numeric", "hint": "e.g. < 1.0"},
    "Current Ratio":     {"type": "numeric", "hint": "e.g. > 1.5"},
    "Quick Ratio":       {"type": "numeric", "hint": "e.g. > 1.0"},
    "Interest Coverage": {"type": "numeric", "hint": "e.g. > 3"},
    "Piotroski":         {"type": "numeric", "hint": "0 to 9  (e.g. >= 7)"},
    # Dividend (values in %)
    "Div Yield":         {"type": "numeric", "hint": "e.g. > 2   (%)"},
    "Payout Ratio":      {"type": "numeric", "hint": "e.g. < 60  (%)"},
    # Technical
    "RSI":               {"type": "numeric", "hint": "e.g. < 40 (oversold) or > 60"},
    "Beta":              {"type": "numeric", "hint": "e.g. < 1.0 (low risk)"},
}

NUMERIC_OPS = [">=", ">", "<=", "<", "==", "!="]
CATEGORICAL_OPS = ["==", "!="]


def load_saved_screens():
    """Load saved screens from JSON file."""
    try:
        if os.path.exists(SAVED_SCREENS_FILE):
            with open(SAVED_SCREENS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_screens_to_file(screens):
    """Persist screens dict to JSON file."""
    try:
        with open(SAVED_SCREENS_FILE, "w") as f:
            json.dump(screens, f, indent=2)
    except Exception:
        pass


def apply_screen_filters(df, conditions, connectors):
    """Apply AND/OR filter conditions to a screener results dataframe."""
    if not conditions or df.empty:
        return df
    masks = []
    for cond in conditions:
        field = cond.get("field")
        op = cond.get("operator")
        value = cond.get("value")
        if field not in df.columns:
            masks.append(pd.Series([True] * len(df), index=df.index))
            continue
        col = df[field]
        field_info = SCREENER_FIELDS.get(field, {"type": "numeric"})
        if field_info["type"] == "categorical":
            if op == "==":
                mask = col == value
            elif op == "!=":
                mask = col != value
            else:
                mask = pd.Series([True] * len(df), index=df.index)
        else:
            col_num = pd.to_numeric(col, errors="coerce")
            try:
                num_val = float(value)
            except (ValueError, TypeError):
                mask = pd.Series([True] * len(df), index=df.index)
            else:
                if op == ">":
                    mask = col_num > num_val
                elif op == ">=":
                    mask = col_num >= num_val
                elif op == "<":
                    mask = col_num < num_val
                elif op == "<=":
                    mask = col_num <= num_val
                elif op == "==":
                    mask = col_num == num_val
                elif op == "!=":
                    mask = col_num != num_val
                else:
                    mask = pd.Series([True] * len(df), index=df.index)
        masks.append(mask.fillna(False))
    if not masks:
        return df
    result_mask = masks[0]
    for i, connector in enumerate(connectors):
        if i + 1 < len(masks):
            if connector == "AND":
                result_mask = result_mask & masks[i + 1]
            else:
                result_mask = result_mask | masks[i + 1]
    return df[result_mask]


def render_screen_builder():
    """Render the Create / Edit Screen panel in the main content area."""
    _edit_mode = st.session_state.get("sb_edit_mode", False)
    _original_name = st.session_state.get("sb_original_name", "")
    if _edit_mode:
        st.subheader(f"Edit Screen: {_original_name}")
    else:
        st.subheader("Create New Screen")
    st.caption("Build a custom filter using AND / OR conditions on screener metrics.")

    screen_name = st.text_input(
        "Screen Name",
        value=st.session_state.get("sb_screen_name", ""),
        placeholder="e.g. High ROE Value Stocks",
        key="sb_screen_name",
    )

    if "sb_conditions" not in st.session_state:
        st.session_state["sb_conditions"] = [
            {"field": "Action", "operator": "==", "value": "BULLISH"}
        ]
    if "sb_connectors" not in st.session_state:
        st.session_state["sb_connectors"] = []

    conditions = st.session_state["sb_conditions"]
    connectors = st.session_state["sb_connectors"]

    st.markdown("#### Filter Conditions")

    to_delete = None
    for i, cond in enumerate(conditions):
        col_conn, col_field, col_op, col_val, col_del = st.columns([1, 2, 1.2, 1.5, 0.5])

        with col_conn:
            if i == 0:
                st.markdown("**WHERE**")
            else:
                while len(connectors) < i:
                    connectors.append("AND")
                conn_val = connectors[i - 1]
                connector = st.selectbox(
                    "",
                    options=["AND", "OR"],
                    index=0 if conn_val == "AND" else 1,
                    key=f"sb_conn_{i}",
                    label_visibility="collapsed",
                )
                connectors[i - 1] = connector

        with col_field:
            field_opts = list(SCREENER_FIELDS.keys())
            cur_field = cond.get("field", field_opts[0])
            field_idx = field_opts.index(cur_field) if cur_field in field_opts else 0
            selected_field = st.selectbox(
                "",
                options=field_opts,
                index=field_idx,
                key=f"sb_field_{i}",
                label_visibility="collapsed",
            )
            conditions[i]["field"] = selected_field

        field_info = SCREENER_FIELDS.get(selected_field, {"type": "numeric"})
        ops = CATEGORICAL_OPS if field_info["type"] == "categorical" else NUMERIC_OPS

        with col_op:
            cur_op = cond.get("operator", ops[0])
            op_idx = ops.index(cur_op) if cur_op in ops else 0
            selected_op = st.selectbox(
                "",
                options=ops,
                index=op_idx,
                key=f"sb_op_{i}",
                label_visibility="collapsed",
            )
            conditions[i]["operator"] = selected_op

        with col_val:
            if field_info["type"] == "categorical":
                cat_vals = field_info.get("values", [])
                cur_val = cond.get("value", cat_vals[0] if cat_vals else "")
                val_idx = cat_vals.index(cur_val) if cur_val in cat_vals else 0
                selected_val = st.selectbox(
                    "",
                    options=cat_vals,
                    index=val_idx,
                    key=f"sb_val_{i}",
                    label_visibility="collapsed",
                )
            else:
                hint = field_info.get("hint", "")
                cur_val = str(cond.get("value", ""))
                selected_val = st.text_input(
                    "",
                    value=cur_val,
                    placeholder=hint,
                    key=f"sb_val_{i}",
                    label_visibility="collapsed",
                )
            conditions[i]["value"] = selected_val

        with col_del:
            if st.button("X", key=f"sb_del_{i}", help="Remove this condition"):
                to_delete = i

    if to_delete is not None:
        conditions.pop(to_delete)
        if connectors:
            if to_delete > 0:
                connectors.pop(to_delete - 1)
            else:
                connectors.pop(0)
        st.session_state["sb_conditions"] = conditions
        st.session_state["sb_connectors"] = connectors
        st.rerun()

    st.session_state["sb_conditions"] = conditions
    st.session_state["sb_connectors"] = connectors

    if st.button("+ Add Condition", key="sb_add_cond"):
        st.session_state["sb_conditions"].append(
            {"field": "Score", "operator": ">=", "value": "0.3"}
        )
        st.session_state["sb_connectors"].append("AND")
        st.rerun()

    st.markdown("---")
    col_save, col_cancel = st.columns([1, 1])
    with col_save:
        if st.button("Save Screen", type="primary", key="sb_save"):
            if not screen_name.strip():
                st.warning("Please enter a name for the screen.")
            else:
                screens = load_saved_screens()
                screens[screen_name.strip()] = {
                    "conditions": list(conditions),
                    "connectors": list(connectors),
                }
                # If renaming, remove the old entry first
                _edit_mode = st.session_state.get("sb_edit_mode", False)
                _orig = st.session_state.get("sb_original_name", "")
                if _edit_mode and _orig and _orig != screen_name.strip() and _orig in screens:
                    del screens[_orig]
                save_screens_to_file(screens)
                st.session_state["saved_screens"] = screens
                st.session_state["sb_edit_mode"] = False
                st.session_state["sb_original_name"] = ""
                # Tell the sidebar to sync the selectbox once on the next rerun
                st.session_state["_sync_to_screen"] = screen_name.strip()
                st.session_state["active_screen"] = screen_name.strip()
                # Store conditions directly so the filter always has them
                st.session_state["active_conds"] = list(conditions)
                st.session_state["active_conns"] = list(connectors)
                st.session_state["show_screen_builder"] = False
                st.success("Screen saved successfully!")
                st.rerun()
    with col_cancel:
        if st.button("Cancel", key="sb_cancel"):
            st.session_state["show_screen_builder"] = False
            st.session_state["sb_edit_mode"] = False
            st.session_state["sb_original_name"] = ""
            st.rerun()

    return conditions, connectors, screen_name.strip()

# =============================================================================
# Section 7c: Bulk Stock Screener
# =============================================================================

def run_screener(stock_dict, market="India"):
    """Run full analysis on multiple stocks, return list of result dicts."""
    results = []
    total = len(stock_dict)
    progress_bar = st.progress(0, text="Screening stocks...")
    status_text = st.empty()

    for idx, (name, ticker) in enumerate(stock_dict.items()):
        status_text.text(f"Analyzing {name} ({idx + 1}/{total})...")
        progress_bar.progress((idx + 1) / total)

        try:
            stock_data = fetch_stock_data(ticker)
            if stock_data["status"] != "ok":
                continue

            info = stock_data.get("info", {})
            sector = get_stock_sector(info)

            # Technical
            technical = calculate_technical_indicators(stock_data["history"])

            # Sentiment
            news = fetch_news_headlines(name, sector, market=market)
            sentiment = analyze_sentiment(news)

            # Fundamental
            fundamental = analyze_fundamentals(
                stock_data.get("quarterly_income"),
                info,
                balance_sheet_df=stock_data.get("quarterly_balance"),
            )

            # Key metrics (for display columns)
            key_metrics = compute_key_metrics(
                info,
                stock_data.get("annual_income"),
                stock_data.get("annual_balance"),
                stock_data.get("cashflow"),
            )

            # Prediction
            prediction = generate_prediction(technical, sentiment, fundamental)
            if prediction["status"] != "ok":
                continue

            # 52-week high/low from price history
            hist = stock_data.get("history")
            w52_high = round(float(hist["High"].max()), 2) if hist is not None and not hist.empty else None
            w52_low  = round(float(hist["Low"].min()),  2) if hist is not None and not hist.empty else None
            cmp_val  = technical.get("current_price")
            pct_from_high = (
                round((cmp_val - w52_high) / w52_high * 100, 1)
                if cmp_val and w52_high else None
            )

            def _pct(val):
                return round(val * 100, 2) if val is not None else None

            results.append({
                # Core
                "Stock": name,
                "Ticker": ticker,
                "Action": prediction["action"],
                "Confidence": prediction["confidence"],
                "Score": prediction["score"],
                "Tech Score": technical.get("score"),
                "Sentiment Score": sentiment.get("score"),
                "Fundamental Score": fundamental.get("score"),
                # Price / Size
                "CMP": cmp_val,
                "Market Cap": key_metrics.get("market_cap"),
                "52W High": w52_high,
                "52W Low": w52_low,
                "% from 52W High": pct_from_high,
                # Valuation
                "P/E": key_metrics.get("pe_ratio"),
                "Fwd P/E": key_metrics.get("forward_pe"),
                "P/B": key_metrics.get("price_to_book"),
                "P/S": key_metrics.get("ps_ratio"),
                "EV/EBITDA": key_metrics.get("ev_ebitda"),
                "EV/Revenue": key_metrics.get("ev_revenue"),
                # Profitability (stored as %)
                "ROE": _pct(key_metrics.get("roe_latest")),
                "ROCE": _pct(key_metrics.get("roce_latest")),
                "ROA": _pct(key_metrics.get("roa")),
                "Gross Margin": _pct(key_metrics.get("gross_margin")),
                "Operating Margin": _pct(key_metrics.get("operating_margin")),
                "Net Margin": _pct(key_metrics.get("net_margin")),
                "EBITDA Margin": _pct(key_metrics.get("ebitda_margin")),
                # Growth (stored as %)
                "Rev Growth YoY": _pct(key_metrics.get("revenue_growth_yoy")),
                "EPS Growth YoY": _pct(key_metrics.get("earnings_growth_yoy")),
                # Financial Health
                "D/E": key_metrics.get("debt_to_equity"),
                "Current Ratio": key_metrics.get("current_ratio"),
                "Quick Ratio": key_metrics.get("quick_ratio"),
                "Interest Coverage": key_metrics.get("interest_coverage"),
                "Piotroski": key_metrics.get("piotroski_score"),
                # Dividend (stored as %)
                "Div Yield": _pct(key_metrics.get("dividend_yield")),
                "Payout Ratio": _pct(key_metrics.get("payout_ratio")),
                # Technical
                "RSI": round(technical.get("rsi_value"), 1) if technical.get("rsi_value") is not None else None,
                "Beta": key_metrics.get("beta"),
                # Per Share
                "EPS": key_metrics.get("eps"),
            })
        except Exception:
            continue  # Skip failed stocks silently

    progress_bar.empty()
    status_text.empty()
    return results


# =============================================================================
# Section 8: Prediction Engine
# =============================================================================

def generate_prediction(technical, sentiment, fundamental, oi_data=None):
    """Generate final Bullish/Neutral/Bearish signal with confidence."""
    components = {}
    weights = {}

    if technical.get("status") == "ok":
        components["technical"] = technical["score"]
        weights["technical"] = WEIGHT_TECHNICAL

    if fundamental.get("status") == "ok":
        components["fundamental"] = fundamental["score"]
        weights["fundamental"] = WEIGHT_FUNDAMENTAL

    if sentiment.get("status") == "ok":
        components["sentiment"] = sentiment["score"]
        weights["sentiment"] = WEIGHT_SENTIMENT

    # OI / Put-Call Ratio adds a small tilt when available
    if oi_data and oi_data.get("status") == "ok":
        components["open_interest"] = oi_data["oi_signal"]
        weights["open_interest"] = 0.08  # small weight; redistributed below

    if not components:
        return {
            "status": "error",
            "message": "No analysis data available",
        }

    # Redistribute weights if some components are missing
    total_weight = sum(weights.values())
    if total_weight > 0:
        for k in weights:
            weights[k] /= total_weight

    # Weighted final score
    final_score = sum(components[k] * weights[k] for k in components)
    final_score = float(np.clip(final_score, -1, 1))

    # Determine signal
    if final_score >= BUY_THRESHOLD:
        action = "BULLISH"
    elif final_score <= SELL_THRESHOLD:
        action = "BEARISH"
    else:
        action = "NEUTRAL"

    # Confidence calculation
    # Base confidence from score magnitude (0-50 points)
    magnitude_conf = min(abs(final_score) / 1.0 * 50, 50)

    # Agreement bonus (0-30 points) - how many components agree with direction
    if len(components) >= 2:
        direction = 1 if final_score >= 0 else -1
        agreeing = sum(1 for v in components.values() if (v >= 0) == (direction >= 0))
        agreement_conf = (agreeing / len(components)) * 30
    else:
        agreement_conf = 10

    # Data source bonus (0-20 points)
    data_conf = (len(components) / 3) * 20

    confidence = min(magnitude_conf + agreement_conf + data_conf, 100)
    confidence = max(confidence, 5)  # Minimum 5%

    return {
        "status": "ok",
        "action": action,
        "score": final_score,
        "confidence": round(confidence, 1),
        "components": components,
        "weights": weights,
    }


# =============================================================================
# Section 8b: Daily Trade Picks & Swing Trade Scanner
# =============================================================================

def _swing_trade_score(technical, fundamental, sentiment, macro_adj):
    """Classify and score every stock — nothing is hidden, all 500 get a label."""
    if technical.get("status") != "ok":
        return 0, "No Data", []

    rsi = technical.get("rsi_value", 50)
    macd_hist = technical.get("macd_hist", 0)
    bb_pct = technical.get("bb_pct", 50)      # 0=at lower band, 100=at upper band
    vol_ratio = technical.get("volume_ratio", 1.0)
    obv_signal = technical.get("obv_signal", 0)
    w52_pct = technical.get("w52_pct", 50)
    score_raw = technical.get("score", 0)
    current_price = technical.get("current_price", 0)
    sma_long = technical.get("sma_long", current_price or 1)
    macd_cross = technical.get("macd_crossover", "")

    reasons = []
    points = 0

    # ── Step 1: Classify setup — priority order matters ──────────────────────

    # OVERBOUGHT: RSI > 75 OR price well above BB upper band
    # Catches: Adani Green (RSI 99), IDBI Bank (RSI 72+BB 80%), Cipla (BB 107%)
    if rsi > 75 or bb_pct > 95:
        setup = "Overbought — Avoid"
        reasons.append(f"RSI {rsi:.0f} overbought" if rsi > 75 else f"BB {bb_pct:.0f}% — above upper band")
        points = max(0, 30 - int((rsi - 75) * 0.5)) if rsi > 75 else 10

    # OVERSOLD REVERSAL: RSI < 35 — potential bottom regardless of OBV
    # Catches: Infosys (RSI 27.4, BB -29%)
    elif rsi < 35:
        setup = "Oversold Reversal"
        reasons.append(f"RSI {rsi:.0f} — deeply oversold, watch for bounce")
        if bb_pct < 10:
            reasons.append(f"BB {bb_pct:.0f}% — at/below lower band (strong support)")
        points = 40 + max(0, int((35 - rsi) * 0.8))  # deeper = higher base score

    # PULLBACK ENTRY: RSI 35–52, price near BB lower half — healthy dip in uptrend
    elif rsi <= 52 and bb_pct <= 40:
        setup = "Pullback Entry"
        reasons.append(f"RSI {rsi:.0f} pulled back, BB {bb_pct:.0f}% near support")
        points = 38

    # MOMENTUM BREAKOUT: RSI 52–70, MACD bullish — trend continuation
    elif 52 < rsi <= 70 and macd_hist > 0:
        setup = "Momentum Breakout"
        reasons.append(f"RSI {rsi:.0f} with MACD bullish — momentum intact")
        points = 40

    # BEARISH: clearly negative technicals
    elif score_raw < -0.2 and macd_cross == "Bearish":
        setup = "Bearish — Avoid"
        reasons.append(f"Tech score {score_raw:+.2f}, MACD bearish crossover")
        points = 15

    # NEUTRAL: everything else
    else:
        setup = "Neutral"
        reasons.append(f"RSI {rsi:.0f}, no clear setup — wait for confirmation")
        points = 20

    # ── Step 2: Volume confirmation bonus (0–20 pts) ─────────────────────────
    if vol_ratio >= 3.0:
        points += 20; reasons.append(f"Strong volume surge {vol_ratio:.1f}x avg")
    elif vol_ratio >= 2.0:
        points += 12; reasons.append(f"High volume {vol_ratio:.1f}x avg")
    elif vol_ratio >= 1.4:
        points += 6; reasons.append(f"Above-avg volume {vol_ratio:.1f}x")
    elif vol_ratio < 0.5:
        points -= 5; reasons.append("Very low volume — low conviction")

    # ── Step 3: MACD alignment (0–12 pts) ────────────────────────────────────
    if macd_cross == "Bullish" and setup in ("Pullback Entry", "Oversold Reversal", "Momentum Breakout"):
        points += 12; reasons.append("MACD bullish crossover confirms entry")
    elif macd_cross == "Bearish" and setup in ("Overbought — Avoid", "Bearish — Avoid"):
        points += 8; reasons.append("MACD bearish crossover confirms caution")

    # ── Step 4: 52-week position context ─────────────────────────────────────
    if w52_pct >= 80 and setup == "Momentum Breakout":
        points += 8; reasons.append(f"Near 52W high ({w52_pct:.0f}%) — strong trend")
    elif w52_pct <= 25 and setup in ("Oversold Reversal", "Pullback Entry"):
        points += 8; reasons.append(f"Near 52W low ({w52_pct:.0f}%) — value zone")

    # ── Step 5: Fundamental quality (0–10 pts) ────────────────────────────────
    if fundamental.get("status") == "ok":
        f_score = fundamental.get("score", 0)
        if f_score > 0.3:
            points += 10; reasons.append("Strong fundamentals back the trade")
        elif f_score > 0.1:
            points += 5; reasons.append("Decent fundamentals")
        elif f_score < -0.2:
            points -= 5; reasons.append("Weak fundamentals — higher risk")

    # ── Step 6: Macro adjustment (−8 to +8) ──────────────────────────────────
    macro_pts = int(macro_adj * 8)
    points += macro_pts
    if macro_adj > 0.05:
        reasons.append("Positive global macro")
    elif macro_adj < -0.05:
        reasons.append("Negative global macro — add caution")

    return min(max(points, 0), 100), setup, reasons


def run_daily_picks(stocks_dict, market="India", max_scan=300):
    """Scan universe for today's best swing/intraday trade setups."""
    # Fetch global macro news once for the full scan
    macro_headlines = fetch_global_macro_news(market)
    macro_sentiment_score = 0.0
    if macro_headlines:
        vader = _get_vader_analyzer()
        macro_scores = [vader.polarity_scores(h["title"])["compound"] for h in macro_headlines if h.get("title")]
        macro_sentiment_score = float(np.mean(macro_scores)) if macro_scores else 0.0

    stocks_list = list(stocks_dict.items())[:max_scan]
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(stocks_list)

    for i, (name, ticker) in enumerate(stocks_list):
        progress_bar.progress((i + 1) / total)
        status_text.text(f"Scanning {i+1}/{total}: {name}")
        try:
            stock_data = fetch_stock_data(ticker)
            if stock_data["status"] != "ok":
                continue
            hist = stock_data.get("history")
            if hist is None or len(hist) < 50:
                continue

            technical = calculate_technical_indicators(hist)
            if technical.get("status") != "ok":
                continue

            info = stock_data.get("info", {})
            fundamental = analyze_fundamentals(
                stock_data.get("quarterly_income"), info,
                balance_sheet_df=stock_data.get("quarterly_balance"),
            )

            # Use cached news for speed (no per-stock news fetch in daily scan)
            sentiment = {"status": "insufficient_data"}

            swing_score, setup, reasons = _swing_trade_score(
                technical, fundamental, sentiment, macro_sentiment_score
            )
            # Show everything — let the user filter. Nothing hidden.

            price = technical.get("current_price", 0)
            action_for_levels = "BEARISH" if setup in ("Overbought — Avoid", "Bearish — Avoid") else "BULLISH"
            trade_lvl = calculate_trade_levels(technical, action_for_levels)

            results.append({
                "Stock": name,
                "Ticker": ticker,
                "Setup": setup,
                "Swing Score": swing_score,
                "Price": round(price, 2),
                "RSI": round(technical.get("rsi_value", 0), 1),
                "MACD": technical.get("macd_crossover", ""),
                "Vol Ratio": round(technical.get("volume_ratio", 1), 2),
                "BB%": technical.get("bb_pct", 50),
                "52W%": technical.get("w52_pct", 50),
                "Tech Score": round(technical.get("score", 0), 3),
                "Target": trade_lvl["target1"] if trade_lvl else None,
                "Stop Loss": trade_lvl["stop_loss"] if trade_lvl else None,
                "R:R": trade_lvl["risk_reward1"] if trade_lvl else None,
                "Reasons": " | ".join(reasons),
            })
        except Exception:
            continue

    progress_bar.empty()
    status_text.empty()

    results.sort(key=lambda x: x["Swing Score"], reverse=True)
    return results, macro_sentiment_score, macro_headlines


def render_daily_picks(results, macro_score, macro_headlines, market="India"):
    """Render daily trade picks with macro context."""
    # Global macro sentiment banner
    macro_color = "#00c853" if macro_score > 0.05 else "#ff1744" if macro_score < -0.05 else "#ff8f00"
    macro_label = "Positive" if macro_score > 0.05 else "Negative" if macro_score < -0.05 else "Neutral"
    st.markdown(
        f"<div style='background:#f5f5f5;border-left:4px solid {macro_color};padding:10px 16px;"
        f"border-radius:4px;margin-bottom:16px;'>"
        f"<b>Global Macro Sentiment:</b> <span style='color:{macro_color};font-weight:700;'>{macro_label}</span>"
        f" &nbsp;|&nbsp; Score: {macro_score:+.3f} &nbsp;|&nbsp; Based on {len(macro_headlines)} global news items"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not results:
        st.warning("No qualifying trade setups found in today's scan.")
        return

    df = pd.DataFrame(results)

    # Summary by setup type
    setup_counts = df["Setup"].value_counts()
    cols = st.columns(len(setup_counts))
    setup_colors = {
        "Momentum Breakout":  "#1976d2",
        "Pullback Entry":     "#00c853",
        "Oversold Reversal":  "#ff9800",
        "Overbought — Avoid": "#b71c1c",
        "Bearish — Avoid":    "#ff1744",
        "Neutral":            "#9e9e9e",
        "No Data":            "#e0e0e0",
    }
    setup_legend = {
        "Momentum Breakout":  "RSI 52-70 + MACD bullish — trend continuing",
        "Pullback Entry":     "RSI 35-52 near BB lower — dip in uptrend",
        "Oversold Reversal":  "RSI < 35 — deeply oversold, watch for bounce",
        "Overbought — Avoid": "RSI > 75 or BB > 95% — stretched, risk of pullback",
        "Bearish — Avoid":    "Negative technicals — avoid or wait",
        "Neutral":            "No clear setup — wait for confirmation",
    }
    for col, (setup, count) in zip(cols, setup_counts.items()):
        color = setup_colors.get(setup, "#9e9e9e")
        legend = setup_legend.get(setup, "")
        col.markdown(
            f"<div style='background:{color}15;border:1px solid {color};border-radius:8px;"
            f"padding:10px;text-align:center;'>"
            f"<b style='color:{color};font-size:1.4em;'>{count}</b><br>"
            f"<span style='font-size:0.82em;font-weight:600;'>{setup}</span><br>"
            f"<span style='font-size:0.72em;color:#666;'>{legend}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Actionable summary
    actionable = df[df["Setup"].isin(["Momentum Breakout", "Pullback Entry", "Oversold Reversal"])]
    avoid = df[df["Setup"].isin(["Overbought — Avoid", "Bearish — Avoid"])]
    neutral = df[df["Setup"].isin(["Neutral"])]
    st.markdown(
        f"**{len(actionable)} stocks worth watching** (Momentum + Pullback + Oversold) | "
        f"**{len(avoid)} stocks to avoid** (Overbought/Bearish) | "
        f"**{len(neutral)} Neutral** (wait for setup)"
    )

    # Setup filter
    setup_options = ["Actionable (Buy setups)", "Avoid (Overbought/Bearish)", "All"] + sorted(df["Setup"].unique().tolist())
    selected_setup = st.radio("Filter by Setup", setup_options, horizontal=True, key="picks_filter")

    if selected_setup == "Actionable (Buy setups)":
        view_df = actionable.copy()
    elif selected_setup == "Avoid (Overbought/Bearish)":
        view_df = avoid.copy()
    elif selected_setup == "All":
        view_df = df.copy()
    else:
        view_df = df[df["Setup"] == selected_setup].copy()

    def _fv(val, fmt=".2f", suffix=""):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        try:
            return f"{val:{fmt}}{suffix}"
        except Exception:
            return "N/A"

    display_df = pd.DataFrame({
        "Stock":      view_df["Stock"],
        "Setup":      view_df["Setup"],
        "Swing Score":view_df["Swing Score"],
        "Price":      view_df["Price"].apply(lambda x: _fv(x, ".2f")),
        "RSI":        view_df["RSI"].apply(lambda x: _fv(x, ".1f")),
        "MACD":       view_df["MACD"],
        "Vol Ratio":  view_df["Vol Ratio"].apply(lambda x: _fv(x, ".2f", "x")),
        "BB%":        view_df["BB%"].apply(lambda x: _fv(x, ".0f", "%")),
        "52W%":       view_df["52W%"].apply(lambda x: _fv(x, ".0f", "%")),
        "Target":     view_df["Target"].apply(lambda x: _fv(x, ".2f")),
        "Stop Loss":  view_df["Stop Loss"].apply(lambda x: _fv(x, ".2f")),
        "R:R":        view_df["R:R"].apply(lambda x: _fv(x, ".2f", "x")),
        "Why":        view_df["Reasons"],
    })

    tbl_h = min(len(display_df) * 38 + 50, 800)
    st.dataframe(
        display_df,
        use_container_width=True,
        height=tbl_h,
        column_config={
            "Swing Score": st.column_config.ProgressColumn(
                "Swing Score", min_value=0, max_value=100, format="%d"
            )
        },
    )
    st.caption(f"Showing {len(display_df)} of {len(df)} stocks scanned | Sorted by Swing Score")

    # Global macro headlines expander
    if macro_headlines:
        with st.expander(f"Global Macro News ({len(macro_headlines)} headlines)"):
            vader = _get_vader_analyzer()
            for h in macro_headlines[:20]:
                title = h.get("title", "")
                source = h.get("source", "")
                sc = vader.polarity_scores(title)["compound"]
                c = "#00c853" if sc > 0.1 else "#ff1744" if sc < -0.1 else "#888"
                st.markdown(
                    f"<span style='color:{c}'>[{sc:+.2f}]</span> {title} "
                    f"<span style='color:gray;font-size:0.82em'>— {source}</span>",
                    unsafe_allow_html=True,
                )


# =============================================================================
# Section 9: UI Helpers
# =============================================================================

def render_gauge_chart(confidence, action):
    """Render a Plotly gauge chart for confidence level."""
    go = _go()
    color_map = {"BULLISH": "#00c853", "BEARISH": "#ff1744", "NEUTRAL": "#ffc107"}
    bar_color = color_map.get(action, "#ffc107")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Confidence Level", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 33], "color": "#ffebee"},
                {"range": [33, 66], "color": "#fff8e1"},
                {"range": [66, 100], "color": "#e8f5e9"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 4},
                "thickness": 0.75,
                "value": confidence,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30))
    return fig


def render_key_metrics(metrics):
    """Render a Key Financial Metrics dashboard section (display-only, does not affect signal)."""
    st.subheader("Key Financial Metrics")

    def _fmt_ratio(val, pct=False):
        if val is None:
            return "N/A"
        if pct:
            return f"{val * 100:.1f}%"
        return f"{val:.2f}"

    def _fmt_mcap(val):
        if val is None:
            return "N/A"
        if val >= 1e12:
            return f"{val / 1e12:.2f}T"
        if val >= 1e9:
            return f"{val / 1e9:.2f}B"
        if val >= 1e7:
            return f"{val / 1e7:.2f}Cr"
        return f"{val:,.0f}"

    # Row 1: P/E, P/B, D/E, Current Ratio
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pe = metrics.get("pe_ratio")
        st.metric("P/E Ratio", f"{pe:.2f}" if pe is not None else "N/A")
    with c2:
        pb = metrics.get("price_to_book")
        st.metric("Price / Book", f"{pb:.2f}" if pb is not None else "N/A")
    with c3:
        de = metrics.get("debt_to_equity")
        st.metric("Debt / Equity", f"{de:.2f}" if de is not None else "N/A")
    with c4:
        cr = metrics.get("current_ratio")
        st.metric("Current Ratio", f"{cr:.2f}" if cr is not None else "N/A")

    # Row 2: ROE, ROCE, Piotroski, Market Cap
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("ROE (Latest)", _fmt_ratio(metrics.get("roe_latest"), pct=True))
        sub_parts = []
        if metrics.get("roe_3yr") is not None:
            sub_parts.append(f"3yr: {metrics['roe_3yr'] * 100:.1f}%")
        if metrics.get("roe_5yr") is not None:
            sub_parts.append(f"5yr: {metrics['roe_5yr'] * 100:.1f}%")
        if sub_parts:
            st.caption(" | ".join(sub_parts))

    with c6:
        st.metric("ROCE (Latest)", _fmt_ratio(metrics.get("roce_latest"), pct=True))
        sub_parts = []
        if metrics.get("roce_3yr") is not None:
            sub_parts.append(f"3yr: {metrics['roce_3yr'] * 100:.1f}%")
        if metrics.get("roce_5yr") is not None:
            sub_parts.append(f"5yr: {metrics['roce_5yr'] * 100:.1f}%")
        if sub_parts:
            st.caption(" | ".join(sub_parts))

    with c7:
        f_score = metrics.get("piotroski_score")
        if f_score is not None:
            if f_score >= 7:
                color = "#00c853"
            elif f_score >= 4:
                color = "#ffc107"
            else:
                color = "#ff1744"
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<span style='font-size:0.875rem;color:rgba(49,51,63,0.6);'>Piotroski F-Score</span><br>"
                f"<span style='font-size:2.25rem;font-weight:700;color:{color};'>{f_score}/9</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.metric("Piotroski F-Score", "N/A")

        details = metrics.get("piotroski_details")
        if details:
            with st.expander("F-Score Breakdown"):
                for criterion, passed in details:
                    icon = "PASS" if passed else "FAIL"
                    c = "#00c853" if passed else "#ff1744"
                    st.markdown(
                        f"<span style='color:{c};font-weight:600;'>[{icon}]</span> {criterion}",
                        unsafe_allow_html=True,
                    )

    with c8:
        st.metric("Market Cap", _fmt_mcap(metrics.get("market_cap")))


def render_screener_results(results):
    """Display screener results with summary cards, filter, and data table."""
    df = pd.DataFrame(results)

    # Part A: Summary cards
    buy_count = len(df[df["Action"] == "BULLISH"])
    hold_count = len(df[df["Action"] == "NEUTRAL"])
    sell_count = len(df[df["Action"] == "BEARISH"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div style='background-color:#e8f5e9;padding:20px;border-radius:10px;text-align:center;'>"
            f"<h2 style='color:#00c853;margin:0;'>{buy_count}</h2>"
            f"<p style='margin:0;color:#2e7d32;'>stocks signalling BULLISH</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div style='background-color:#fff8e1;padding:20px;border-radius:10px;text-align:center;'>"
            f"<h2 style='color:#ff8f00;margin:0;'>{hold_count}</h2>"
            f"<p style='margin:0;color:#f57f17;'>stocks signalling NEUTRAL</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div style='background-color:#ffebee;padding:20px;border-radius:10px;text-align:center;'>"
            f"<h2 style='color:#ff1744;margin:0;'>{sell_count}</h2>"
            f"<p style='margin:0;color:#c62828;'>stocks signalling BEARISH</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Part B: Filter tabs
    filter_action = st.radio(
        "Filter by Signal",
        options=["All", "BULLISH", "NEUTRAL", "BEARISH"],
        horizontal=True,
    )

    if filter_action != "All":
        filtered_df = df[df["Action"] == filter_action].copy()
    else:
        filtered_df = df.copy()

    # Sort by confidence descending
    filtered_df = filtered_df.sort_values("Confidence", ascending=False).reset_index(drop=True)

    # ---- Shared helpers ----
    def _fv(val, fmt=".2f", suffix=""):
        """Format a numeric value safely."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        try:
            return f"{val:{fmt}}{suffix}"
        except Exception:
            return "N/A"

    def _fmt_mcap(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        if val >= 1e12:
            return f"{val / 1e12:.2f}T"
        if val >= 1e9:
            return f"{val / 1e9:.2f}B"
        if val >= 1e7:
            return f"{val / 1e7:.2f}Cr"
        return f"{val:,.0f}"

    tbl_h = min(len(filtered_df) * 38 + 40, 800)
    conf_col = st.column_config.ProgressColumn(
        "Confidence (%)", min_value=0, max_value=100, format="%.1f%%"
    )

    tab_main, tab_val, tab_prof, tab_growth, tab_health, tab_div, tab_tech = st.tabs([
        "Overview", "Valuation", "Profitability", "Growth", "Financial Health", "Dividend", "Technical"
    ])

    # ---- Tab 1: Overview ----
    with tab_main:
        t1 = pd.DataFrame({
            "Stock":          filtered_df["Stock"],
            "Action":         filtered_df["Action"],
            "Confidence (%)": filtered_df["Confidence"],
            "Score":          filtered_df["Score"].apply(lambda x: _fv(x, "+.3f")),
            "CMP":            filtered_df["CMP"].apply(lambda x: _fv(x, ".2f")),
            "Market Cap":     filtered_df["Market Cap"].apply(_fmt_mcap),
            "52W High":       filtered_df["52W High"].apply(lambda x: _fv(x, ".2f")),
            "52W Low":        filtered_df["52W Low"].apply(lambda x: _fv(x, ".2f")),
            "% from 52W High":filtered_df["% from 52W High"].apply(lambda x: _fv(x, ".1f", "%")),
            "Piotroski":      filtered_df["Piotroski"].apply(lambda x: f"{int(x)}/9" if pd.notna(x) and x is not None else "N/A"),
            "EPS":            filtered_df["EPS"].apply(lambda x: _fv(x, ".2f")),
        })
        st.dataframe(t1, use_container_width=True, height=tbl_h,
                     column_config={"Confidence (%)": conf_col})

    # ---- Tab 2: Valuation ----
    with tab_val:
        t2 = pd.DataFrame({
            "Stock":     filtered_df["Stock"],
            "CMP":       filtered_df["CMP"].apply(lambda x: _fv(x, ".2f")),
            "P/E":       filtered_df["P/E"].apply(lambda x: _fv(x, ".2f")),
            "Fwd P/E":   filtered_df["Fwd P/E"].apply(lambda x: _fv(x, ".2f")),
            "P/B":       filtered_df["P/B"].apply(lambda x: _fv(x, ".2f")),
            "P/S":       filtered_df["P/S"].apply(lambda x: _fv(x, ".2f")),
            "EV/EBITDA": filtered_df["EV/EBITDA"].apply(lambda x: _fv(x, ".2f")),
            "EV/Revenue":filtered_df["EV/Revenue"].apply(lambda x: _fv(x, ".2f")),
            "Market Cap":filtered_df["Market Cap"].apply(_fmt_mcap),
        })
        st.dataframe(t2, use_container_width=True, height=tbl_h)

    # ---- Tab 3: Profitability ----
    with tab_prof:
        t3 = pd.DataFrame({
            "Stock":           filtered_df["Stock"],
            "ROE (%)":         filtered_df["ROE"].apply(lambda x: _fv(x, ".1f", "%")),
            "ROCE (%)":        filtered_df["ROCE"].apply(lambda x: _fv(x, ".1f", "%")),
            "ROA (%)":         filtered_df["ROA"].apply(lambda x: _fv(x, ".1f", "%")),
            "Gross Margin (%)":filtered_df["Gross Margin"].apply(lambda x: _fv(x, ".1f", "%")),
            "Oper Margin (%)": filtered_df["Operating Margin"].apply(lambda x: _fv(x, ".1f", "%")),
            "Net Margin (%)":  filtered_df["Net Margin"].apply(lambda x: _fv(x, ".1f", "%")),
            "EBITDA Margin (%)":filtered_df["EBITDA Margin"].apply(lambda x: _fv(x, ".1f", "%")),
        })
        st.dataframe(t3, use_container_width=True, height=tbl_h)

    # ---- Tab 4: Growth ----
    with tab_growth:
        t4 = pd.DataFrame({
            "Stock":            filtered_df["Stock"],
            "Rev Growth YoY (%)":filtered_df["Rev Growth YoY"].apply(lambda x: _fv(x, ".1f", "%")),
            "EPS Growth YoY (%)":filtered_df["EPS Growth YoY"].apply(lambda x: _fv(x, ".1f", "%")),
            "EPS":              filtered_df["EPS"].apply(lambda x: _fv(x, ".2f")),
            "Score":            filtered_df["Score"].apply(lambda x: _fv(x, "+.3f")),
        })
        st.dataframe(t4, use_container_width=True, height=tbl_h)

    # ---- Tab 5: Financial Health ----
    with tab_health:
        t5 = pd.DataFrame({
            "Stock":             filtered_df["Stock"],
            "D/E":               filtered_df["D/E"].apply(lambda x: _fv(x, ".2f")),
            "Current Ratio":     filtered_df["Current Ratio"].apply(lambda x: _fv(x, ".2f")),
            "Quick Ratio":       filtered_df["Quick Ratio"].apply(lambda x: _fv(x, ".2f")),
            "Interest Coverage": filtered_df["Interest Coverage"].apply(lambda x: _fv(x, ".1f")),
            "Piotroski":         filtered_df["Piotroski"].apply(lambda x: f"{int(x)}/9" if pd.notna(x) and x is not None else "N/A"),
            "Fundamental Score": filtered_df["Fundamental Score"].apply(lambda x: _fv(x, "+.3f")),
        })
        st.dataframe(t5, use_container_width=True, height=tbl_h)

    # ---- Tab 6: Dividend ----
    with tab_div:
        t6 = pd.DataFrame({
            "Stock":         filtered_df["Stock"],
            "Div Yield (%)": filtered_df["Div Yield"].apply(lambda x: _fv(x, ".2f", "%")),
            "Payout Ratio (%)":filtered_df["Payout Ratio"].apply(lambda x: _fv(x, ".1f", "%")),
            "EPS":           filtered_df["EPS"].apply(lambda x: _fv(x, ".2f")),
            "P/E":           filtered_df["P/E"].apply(lambda x: _fv(x, ".2f")),
        })
        st.dataframe(t6, use_container_width=True, height=tbl_h)

    # ---- Tab 7: Technical ----
    with tab_tech:
        t7 = pd.DataFrame({
            "Stock":           filtered_df["Stock"],
            "CMP":             filtered_df["CMP"].apply(lambda x: _fv(x, ".2f")),
            "RSI":             filtered_df["RSI"].apply(lambda x: _fv(x, ".1f")),
            "Beta":            filtered_df["Beta"].apply(lambda x: _fv(x, ".2f")),
            "52W High":        filtered_df["52W High"].apply(lambda x: _fv(x, ".2f")),
            "52W Low":         filtered_df["52W Low"].apply(lambda x: _fv(x, ".2f")),
            "% from 52W High": filtered_df["% from 52W High"].apply(lambda x: _fv(x, ".1f", "%")),
            "Tech Score":      filtered_df["Tech Score"].apply(lambda x: _fv(x, "+.3f")),
            "Sentiment Score": filtered_df["Sentiment Score"].apply(lambda x: _fv(x, "+.3f")),
        })
        st.dataframe(t7, use_container_width=True, height=tbl_h)

    st.caption(f"Showing {len(filtered_df)} of {len(df)} stocks")


def render_component_breakdown(technical, sentiment, fundamental, prediction):
    """Render 3-column breakdown of each analysis component."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Technical Analysis")
        if technical.get("status") == "ok":
            score = technical["score"]
            color = "#00c853" if score > 0.1 else "#ff1744" if score < -0.1 else "#ffc107"
            st.markdown(f"**Score:** <span style='color:{color}'>{score:+.3f}</span>", unsafe_allow_html=True)
            st.markdown(f"- Price: **{technical['current_price']:.2f}**")
            st.markdown(f"- RSI ({RSI_PERIOD}): **{technical['rsi_value']:.1f}**")
            rsi_label = "Overbought" if technical["rsi_value"] > 70 else "Oversold" if technical["rsi_value"] < 30 else "Neutral"
            st.markdown(f"- RSI Zone: **{rsi_label}**")
            st.markdown(f"- SMA{SMA_SHORT}: {technical['sma_short']:.2f} | SMA{SMA_LONG}: {technical['sma_long']:.2f}")
            sma_label = "Bullish" if technical["sma_signal"] > 0 else "Bearish"
            st.markdown(f"- SMA Cross: **{sma_label}**")
            if technical.get("macd_crossover"):
                st.markdown(f"- MACD: **{technical['macd_crossover']}** (hist: {technical.get('macd_hist', 0):+.3f})")
            if technical.get("bb_pct") is not None:
                st.markdown(f"- BB Position: **{technical['bb_pct']}%** (0=lower, 100=upper band)")
            obv_t = technical.get("obv_trend", "N/A")
            vol_r = technical.get("volume_ratio", 1.0)
            st.markdown(f"- OBV Trend: **{obv_t}** | Vol Ratio: **{vol_r:.1f}x**")
            mom_label = "Positive" if technical["momentum_signal"] > 0.1 else "Negative" if technical["momentum_signal"] < -0.1 else "Flat"
            st.markdown(f"- Momentum: **{mom_label}**")
            st.markdown(f"- ATR: {technical.get('atr', 0):.2f} | 52W: {technical.get('w52_low', 0):.0f}–{technical.get('w52_high', 0):.0f}")
            st.markdown(f"- 52W Position: **{technical.get('w52_pct', 0):.0f}%**")
        else:
            st.warning("Insufficient price data")

    with col2:
        st.subheader("Sentiment Analysis")
        if sentiment.get("status") == "ok":
            score = sentiment["score"]
            color = "#00c853" if score > 0.1 else "#ff1744" if score < -0.1 else "#ffc107"
            st.markdown(f"**Score:** <span style='color:{color}'>{score:+.3f}</span>", unsafe_allow_html=True)
            st.markdown(f"- Stock News: {sentiment['stock_sentiment']:+.3f}")
            st.markdown(f"- Sector News: {sentiment['sector_sentiment']:+.3f}")
            st.markdown(f"- Market News: {sentiment['market_sentiment']:+.3f}")
            st.markdown(f"- Headlines Analyzed: **{sentiment['headline_count']}**")
        else:
            st.warning("No news data available")

    with col3:
        st.subheader("Fundamental Analysis")
        if fundamental.get("status") == "ok":
            score = fundamental["score"]
            color = "#00c853" if score > 0.1 else "#ff1744" if score < -0.1 else "#ffc107"
            st.markdown(f"**Score:** <span style='color:{color}'>{score:+.3f}</span>", unsafe_allow_html=True)

            if fundamental.get("revenue_growth") is not None:
                st.markdown(f"- Revenue Growth (QoQ): {fundamental['revenue_growth']:+.3f}")
            else:
                st.markdown("- Revenue Growth: N/A")

            if fundamental.get("raw_margin") is not None:
                st.markdown(f"- Profit Margin: {fundamental['raw_margin']*100:.1f}%")
            else:
                st.markdown("- Profit Margin: N/A")

            if fundamental.get("profit_growth") is not None:
                st.markdown(f"- Profit Growth (QoQ): {fundamental['profit_growth']:+.3f}")
            else:
                st.markdown("- Profit Growth: N/A")

            if fundamental.get("raw_de_ratio") is not None:
                st.markdown(f"- Debt/Equity Ratio: {fundamental['raw_de_ratio']:.2f}")
            else:
                st.markdown("- Debt/Equity Ratio: N/A")

            if fundamental.get("raw_current_ratio") is not None:
                st.markdown(f"- Current Ratio: {fundamental['raw_current_ratio']:.2f}")
            else:
                st.markdown("- Current Ratio: N/A")

            if fundamental.get("raw_roe") is not None:
                st.markdown(f"- Return on Equity: {fundamental['raw_roe']*100:.1f}%")
            else:
                st.markdown("- Return on Equity: N/A")

            if fundamental.get("raw_roca") is not None:
                st.markdown(f"- Return on Capital Assets: {fundamental['raw_roca']*100:.1f}%")
            else:
                st.markdown("- ROCA: N/A")

            if fundamental.get("raw_roce") is not None:
                st.markdown(f"- Return on Capital Employed: {fundamental['raw_roce']*100:.1f}%")
            else:
                st.markdown("- ROCE: N/A")

            st.markdown(f"- Signals Available: **{fundamental['available_signals']}/8**")
        else:
            st.warning("Insufficient fundamental data")


def render_news_table(sentiment_result):
    """Render color-coded news headlines with sentiment scores."""
    if sentiment_result.get("status") != "ok":
        st.info("No news headlines available.")
        return

    news_data = sentiment_result.get("news_with_scores", {})

    for category, label in [("stock", "Stock News"), ("sector", "Sector News"), ("market", "Market News")]:
        items = news_data.get(category, [])
        if not items:
            continue

        st.markdown(f"**{label}**")
        for item in items:
            title = item.get("title", "")
            score = item.get("sentiment_score", 0)
            source = item.get("source", "Unknown")

            if score >= 0.2:
                emoji = "+"
                color = "#00c853"
            elif score <= -0.2:
                emoji = "-"
                color = "#ff1744"
            else:
                emoji = "~"
                color = "#ffc107"

            st.markdown(
                f"<span style='color:{color}'>[{emoji} {score:+.2f}]</span> {title} "
                f"<span style='color:gray;font-size:0.85em'>- {source}</span>",
                unsafe_allow_html=True,
            )
        st.markdown("---")


def render_price_chart(history, stock_name, market="India"):
    """Render stock price chart with SMAs, Bollinger Bands, MACD, and Volume."""
    go = _go()
    make_subplots = _make_subplots()

    close = history["Close"]
    high = history["High"]
    low = history["Low"]
    volume = history.get("Volume", pd.Series(dtype=float))

    sma_short = close.rolling(window=SMA_SHORT).mean()
    sma_long = close.rolling(window=SMA_LONG).mean()

    # Bollinger Bands
    bb_mid = close.rolling(window=BB_PERIOD).mean()
    bb_std_col = close.rolling(window=BB_PERIOD).std()
    bb_upper = bb_mid + BB_STD * bb_std_col
    bb_lower = bb_mid - BB_STD * bb_std_col

    # MACD
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    macd_hist = macd_line - signal_line

    has_volume = volume is not None and not volume.empty and volume.sum() > 0

    rows = 3 if has_volume else 2
    row_heights = [0.55, 0.25, 0.20] if has_volume else [0.65, 0.35]
    subplot_titles = [f"{stock_name} - Price & Indicators", "MACD"] + (["Volume"] if has_volume else [])

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Row 1: Candlestick + SMAs + Bollinger Bands
    fig.add_trace(go.Candlestick(
        x=history.index, open=history["Open"], high=high, low=low, close=close,
        name="Price", increasing_line_color="#00c853", decreasing_line_color="#ff1744",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=sma_short, name=f"SMA{SMA_SHORT}",
                             line=dict(color="#ff9800", width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=sma_long, name=f"SMA{SMA_LONG}",
                             line=dict(color="#e91e63", width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=bb_upper, name="BB Upper",
                             line=dict(color="rgba(100,100,200,0.5)", width=1, dash="dot"),
                             showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=bb_lower, name="BB Lower",
                             line=dict(color="rgba(100,100,200,0.5)", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(100,100,200,0.05)",
                             showlegend=False), row=1, col=1)

    # Row 2: MACD
    colors_hist = ["#00c853" if v >= 0 else "#ff1744" for v in macd_hist]
    fig.add_trace(go.Bar(x=history.index, y=macd_hist, name="MACD Hist",
                         marker_color=colors_hist, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=macd_line, name="MACD",
                             line=dict(color="#1976d2", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=signal_line, name="Signal",
                             line=dict(color="#ff9800", width=1, dash="dash")), row=2, col=1)

    # Row 3: Volume
    if has_volume:
        vol_colors = ["#00c853" if c >= o else "#ff1744"
                      for c, o in zip(history["Close"], history["Open"])]
        fig.add_trace(go.Bar(x=history.index, y=volume, name="Volume",
                             marker_color=vol_colors, showlegend=False), row=3, col=1)
        vol_ma = volume.rolling(window=VOLUME_MA_PERIOD).mean()
        fig.add_trace(go.Scatter(x=history.index, y=vol_ma, name=f"Vol MA{VOLUME_MA_PERIOD}",
                                 line=dict(color="#9c27b0", width=1)), row=3, col=1)

    currency = "USD" if market == "US" else "INR"
    fig.update_layout(
        height=600,
        margin=dict(t=60, b=40, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text=f"Price ({currency})", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    if has_volume:
        fig.update_yaxes(title_text="Volume", row=3, col=1)

    return fig


def render_trade_recommendation(prediction, trade_levels, oi_data, technical):
    """Render actionable trade setup: entry, target, stop loss, risk-reward."""
    action = prediction["action"]
    score = prediction["score"]
    confidence = prediction["confidence"]

    action_colors = {"BULLISH": "#00c853", "BEARISH": "#ff1744", "NEUTRAL": "#ffc107"}
    action_bg = {"BULLISH": "#e8f5e9", "BEARISH": "#ffebee", "NEUTRAL": "#fff8e1"}
    action_border = action_colors.get(action, "#ffc107")
    bg = action_bg.get(action, "#fff8e1")

    # Header signal box
    st.markdown(
        f"""
        <div style="border:2px solid {action_border};background:{bg};border-radius:10px;
                    padding:16px 20px;margin-bottom:16px;">
          <h2 style="margin:0;color:{action_border};">{action}</h2>
          <p style="margin:4px 0 0 0;font-size:1em;color:#555;">
            Composite Score: <b>{score:+.3f}</b> &nbsp;|&nbsp; Confidence: <b>{confidence:.1f}%</b>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Trade levels card
    if trade_levels:
        col_entry, col_target, col_sl = st.columns(3)
        with col_entry:
            st.markdown("**Entry Zone**")
            st.markdown(f"₹ {trade_levels['entry_low']:.2f} – {trade_levels['entry_high']:.2f}")
            st.caption(f"Support: ₹{trade_levels['nearest_support']:.2f}")
        with col_target:
            st.markdown("**Targets**")
            st.markdown(f"T1: ₹{trade_levels['target1']:.2f}  &nbsp; (R:R = {trade_levels['risk_reward1']}x)")
            st.markdown(f"T2: ₹{trade_levels['target2']:.2f}  &nbsp; (R:R = {trade_levels['risk_reward2']}x)")
            st.caption(f"T3 (extended): ₹{trade_levels['target3']:.2f}")
        with col_sl:
            st.markdown("**Stop Loss**")
            sl_pct = abs(trade_levels["stop_loss"] - trade_levels["entry_low"]) / trade_levels["entry_low"] * 100
            st.markdown(f"₹{trade_levels['stop_loss']:.2f}  &nbsp; ({sl_pct:.1f}% risk)")
            st.caption(f"Risk/share: ₹{trade_levels['risk_per_share']:.2f} | ATR: ₹{technical.get('atr', 0):.2f}")

    # OI data row
    if oi_data and oi_data.get("status") == "ok":
        pcr = oi_data.get("pcr", 0)
        pcr_label = "Bullish (heavy hedging)" if pcr > 1.2 else "Bearish (call overwrite)" if pcr < 0.8 else "Neutral"
        call_oi = oi_data.get("call_oi", 0)
        put_oi = oi_data.get("put_oi", 0)
        st.markdown(
            f"**Open Interest** — PCR: **{pcr:.3f}** ({pcr_label}) | "
            f"Call OI: {call_oi:,} | Put OI: {put_oi:,}",
        )


# =============================================================================
# Section 10: Main App
# =============================================================================

def _show_disclaimer_modal():
    """Show a full-page disclaimer that must be acknowledged before using the app."""
    st.markdown(
        """
        <style>
        .disclaimer-box {
            border: 2px solid #ff6f00;
            border-radius: 12px;
            padding: 30px;
            background-color: #fffde7;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Important Disclaimer — Please Read Before Continuing")
    st.markdown(
        """
        <div class="disclaimer-box">

        <h3 style="color:#e65100;">This is an Educational Tool — NOT Financial Advice</h3>

        By clicking <strong>"I Understand and Accept"</strong> below, you confirm that you have read and
        understood the following:

        <ol>
          <li><strong>Not a SEBI-Registered Investment Adviser:</strong> This platform and its creator are
          <u>not registered</u> with the Securities and Exchange Board of India (SEBI) as an Investment Adviser
          under the SEBI (Investment Advisers) Regulations, 2013.</li>

          <li><strong>Not an SEC-Registered Investment Adviser:</strong> This platform and its creator are
          <u>not registered</u> with the U.S. Securities and Exchange Commission (SEC) under the Investment
          Advisers Act of 1940.</li>

          <li><strong>No Buy/Sell/Hold Recommendations:</strong> The signals displayed (Bullish / Neutral /
          Bearish) are <u>output scores from mathematical models only</u> — they are <strong>NOT</strong>
          recommendations or advice to buy, sell, or hold any security.</li>

          <li><strong>Educational & Research Purpose Only:</strong> This tool is built to share the creator's
          personal understanding of publicly available data analysis techniques. It is strictly for learning
          and research purposes.</li>

          <li><strong>Data Accuracy Not Guaranteed:</strong> Market data is sourced from third-party providers
          (Yahoo Finance via yfinance). Data may be delayed, inaccurate, incomplete, or unavailable. No
          warranty is made regarding its accuracy or timeliness.</li>

          <li><strong>Past Performance Is Not Indicative of Future Results:</strong> Technical, sentiment, and
          fundamental signals are based on historical data and do not predict future market movements.</li>

          <li><strong>Invest at Your Own Risk:</strong> Financial markets carry significant risk of capital
          loss. Always consult a <u>SEBI-registered Investment Adviser</u> or a <u>licensed financial
          professional</u> before making any investment decision.</li>

          <li><strong>No Liability:</strong> The creator of this tool bears no responsibility or liability
          for any financial decisions made based on the output of this platform.</li>
        </ol>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_btn, col_right = st.columns([2, 2, 2])
    with col_btn:
        if st.button(
            "I Understand and Accept",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["disclaimer_accepted"] = True
            st.rerun()

    st.stop()


def main():
    st.set_page_config(
        page_title="Stock Market Analysis Tool",
        page_icon="📈",
        layout="wide",
    )

    # Send an immediate empty write so the WebSocket has traffic right away.
    # Without this, Railway's reverse proxy can drop the WebSocket connection
    # before Streamlit finishes building the page (Railway idle-timeout ~2 min).
    st.write("")

    # Inject Material Symbols font so sidebar icons render correctly on all hosts
    st.markdown(
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap" />',
        unsafe_allow_html=True,
    )

    # Show disclaimer modal on first visit (session-scoped)
    if not st.session_state.get("disclaimer_accepted"):
        _show_disclaimer_modal()

    # Sidebar
    with st.sidebar:
        market = st.radio("Market", ["India", "US"], horizontal=True)

        if market == "India":
            st.header("Indian Stock Market")
            input_mode = st.radio(
                "How do you want to select a stock?",
                options=["Browse by Category", "Browse by Sector", "Enter Custom Ticker", "Stock Screener", "Daily Trade Picks"],
                horizontal=True,
            )

            # Map category names to NSE index keys for live fetching
            _NSE_INDEX_KEY = {
                "Nifty Next 50": "niftynext50",
                "Mid Cap": "midcap150",
                "Small Cap": "smallcap250",
                "Micro Cap": "microcap250",
            }

            if input_mode == "Browse by Category":
                category_options = list(STOCK_CATEGORIES.keys()) + ["All NSE Stocks (~2,600)"]
                category = st.selectbox(
                    "Select Category",
                    options=category_options,
                )
                if category == "All NSE Stocks (~2,600)":
                    with st.spinner("Fetching full NSE stock list..."):
                        stock_dict = fetch_all_nse_stocks()
                    if not stock_dict:
                        st.warning("Could not fetch NSE list. Showing curated stocks instead.")
                        all_stocks = {}
                        for cat_dict in STOCK_CATEGORIES.values():
                            for n, t in cat_dict.items():
                                if t not in all_stocks.values():
                                    all_stocks[n] = t
                        stock_dict = all_stocks
                elif category in _NSE_INDEX_KEY:
                    idx_key = _NSE_INDEX_KEY[category]
                    with st.spinner(f"Fetching live NSE {category} constituents..."):
                        stock_dict = fetch_nse_index_constituents(idx_key)
                    if not stock_dict:
                        st.caption("Live fetch failed — using curated list as fallback.")
                        stock_dict = STOCK_CATEGORIES[category]
                    else:
                        st.caption(f"Live data: {len(stock_dict)} stocks from NSE")
                else:
                    stock_dict = STOCK_CATEGORIES[category]
                stock_names = sorted(stock_dict.keys())
                default_idx = (
                    stock_names.index("Reliance Industries")
                    if category == "Large Cap (Nifty 50)" and "Reliance Industries" in stock_names
                    else 0
                )
                selected_stock = st.selectbox(
                    "Select Stock",
                    options=stock_names,
                    index=default_idx,
                )
                ticker = stock_dict[selected_stock]
                stock_name = selected_stock
            elif input_mode == "Browse by Sector":
                sector_name = st.selectbox("Select Sector", options=sorted(STOCK_SECTORS.keys()))
                sector_stocks = STOCK_SECTORS[sector_name]
                stock_names = sorted(sector_stocks.keys())
                selected_stock = st.selectbox("Select Stock", options=stock_names)
                ticker = sector_stocks[selected_stock]
                stock_name = selected_stock
            elif input_mode == "Enter Custom Ticker":
                st.markdown("Enter any NSE stock symbol or company name (e.g. `ZOMATO`, `Tata Motors`)")
                custom_ticker = st.text_input(
                    "NSE Symbol/Name",
                    value="",
                    placeholder="e.g. ZOMATO or Tata Motors",
                ).strip()
                if custom_ticker:
                    ticker, stock_name = resolve_ticker(custom_ticker, market="NSE")
                else:
                    ticker = None
                    stock_name = None
            elif input_mode == "Daily Trade Picks":
                ticker = None
                stock_name = None
            else:
                # Stock Screener mode
                ticker = None
                stock_name = None
        else:
            # US Market
            st.header("US Stock Market")
            input_mode = st.radio(
                "How do you want to select a stock?",
                options=["Browse by Index", "Browse by Sector", "Enter Custom Ticker", "Stock Screener"],
                horizontal=True,
            )

            if input_mode == "Browse by Index":
                index_name = st.selectbox(
                    "Select Index",
                    options=["S&P 500", "NASDAQ 100", "Dow 30", "NYSE (~2,700)", "NASDAQ (~4,000)"],
                )
                with st.spinner("Loading index constituents..."):
                    if index_name == "S&P 500":
                        stock_dict = fetch_sp500_stocks()
                    elif index_name == "NASDAQ 100":
                        stock_dict = fetch_nasdaq100_stocks()
                    elif index_name == "Dow 30":
                        stock_dict = fetch_dow30_stocks()
                    elif index_name == "NYSE (~2,700)":
                        stock_dict = fetch_nyse_stocks()
                    else:
                        stock_dict = fetch_nasdaq_stocks()

                if not stock_dict:
                    st.warning(f"Could not fetch {index_name} constituents. Showing curated US stocks instead.")
                    all_stocks = {}
                    for cat_dict in US_STOCK_SECTORS.values():
                        for n, t in cat_dict.items():
                            if t not in all_stocks.values():
                                all_stocks[n] = t
                    stock_dict = all_stocks
                
                stock_names = sorted(stock_dict.keys())
                selected_stock = st.selectbox("Select Stock", options=stock_names)
                ticker = stock_dict[selected_stock]
                stock_name = selected_stock
            elif input_mode == "Browse by Sector":
                sector_name = st.selectbox("Select Sector", options=sorted(US_STOCK_SECTORS.keys()))
                sector_stocks = US_STOCK_SECTORS[sector_name]
                stock_names = sorted(sector_stocks.keys())
                selected_stock = st.selectbox("Select Stock", options=stock_names)
                ticker = sector_stocks[selected_stock]
                stock_name = selected_stock
            elif input_mode == "Enter Custom Ticker":
                st.markdown("Enter any US stock ticker or company name (e.g. `AAPL`, `Apple`)")
                custom_ticker = st.text_input(
                    "US Ticker/Name",
                    value="",
                    placeholder="e.g. AAPL or Apple Inc",
                ).strip()
                if custom_ticker:
                    ticker, stock_name = resolve_ticker(custom_ticker, market="US")
                else:
                    ticker = None
                    stock_name = None
            else:
                # Stock Screener mode
                ticker = None
                stock_name = None

        st.markdown("---")

        # Screener mode variables
        screener_mode = input_mode == "Stock Screener"
        daily_picks_mode = (market == "India" and input_mode == "Daily Trade Picks")
        screener_stocks = {}
        screener_btn = False
        analyze_btn = False
        daily_picks_btn = False

        if screener_mode:
            if market == "India":
                scope_options = (
                    ["All NSE Stocks", "Nifty 500 (live)", "Nifty Midcap 150 (live)",
                     "Nifty Smallcap 250 (live)", "Nifty Microcap 250 (live)", "Curated Stocks"]
                    + list(STOCK_CATEGORIES.keys())
                    + sorted(STOCK_SECTORS.keys())
                )
                screener_scope = st.selectbox("Screener Scope", options=scope_options)

                live_scope_map = {
                    "Nifty 500 (live)": "nifty500",
                    "Nifty Midcap 150 (live)": "midcap150",
                    "Nifty Smallcap 250 (live)": "smallcap250",
                    "Nifty Microcap 250 (live)": "microcap250",
                }

                if screener_scope == "All NSE Stocks":
                    with st.spinner("Fetching NSE stock list..."):
                        screener_stocks = fetch_all_nse_stocks()
                    if not screener_stocks:
                        st.warning("Could not fetch NSE stock list. Falling back to curated stocks.")
                        all_stocks = {}
                        for cat_dict in STOCK_CATEGORIES.values():
                            for name, tick in cat_dict.items():
                                if tick not in all_stocks.values():
                                    all_stocks[name] = tick
                        screener_stocks = all_stocks
                elif screener_scope in live_scope_map:
                    with st.spinner(f"Fetching {screener_scope}..."):
                        screener_stocks = fetch_nse_index_constituents(live_scope_map[screener_scope])
                    if not screener_stocks:
                        st.warning("Live fetch failed. Try again.")
                elif screener_scope == "Curated Stocks":
                    all_stocks = {}
                    for cat_dict in STOCK_CATEGORIES.values():
                        for name, tick in cat_dict.items():
                            if tick not in all_stocks.values():
                                all_stocks[name] = tick
                    screener_stocks = all_stocks
                elif screener_scope in STOCK_CATEGORIES:
                    idx_key = _NSE_INDEX_KEY.get(screener_scope)
                    if idx_key:
                        with st.spinner(f"Fetching live {screener_scope}..."):
                            screener_stocks = fetch_nse_index_constituents(idx_key)
                        if not screener_stocks:
                            screener_stocks = STOCK_CATEGORIES[screener_scope]
                    else:
                        screener_stocks = STOCK_CATEGORIES[screener_scope]
                elif screener_scope in STOCK_SECTORS:
                    screener_stocks = STOCK_SECTORS[screener_scope]
            else:
                # US screener
                scope_options = (
                    ["S&P 500", "NASDAQ 100", "Dow 30", "NYSE (~2,700)", "NASDAQ (~4,000)"]
                    + sorted(US_STOCK_SECTORS.keys())
                )
                screener_scope = st.selectbox("Screener Scope", options=scope_options)

                if screener_scope == "S&P 500":
                    with st.spinner("Fetching S&P 500 list..."):
                        screener_stocks = fetch_sp500_stocks()
                elif screener_scope == "NASDAQ 100":
                    with st.spinner("Fetching NASDAQ 100 list..."):
                        screener_stocks = fetch_nasdaq100_stocks()
                elif screener_scope == "Dow 30":
                    with st.spinner("Fetching Dow 30 list..."):
                        screener_stocks = fetch_dow30_stocks()
                elif screener_scope == "NYSE (~2,700)":
                    with st.spinner("Fetching NYSE stock list..."):
                        screener_stocks = fetch_nyse_stocks()
                elif screener_scope == "NASDAQ (~4,000)":
                    with st.spinner("Fetching NASDAQ stock list..."):
                        screener_stocks = fetch_nasdaq_stocks()
                elif screener_scope in US_STOCK_SECTORS:
                    screener_stocks = US_STOCK_SECTORS[screener_scope]

                if not screener_stocks:
                    st.warning("Could not fetch stock list. Try again later.")

            st.markdown(f"**Stocks to scan:** {len(screener_stocks)}")

            # --- Saved Screens ---
            if "saved_screens" not in st.session_state:
                st.session_state["saved_screens"] = load_saved_screens()
            saved_screens = st.session_state["saved_screens"]

            if saved_screens:
                screen_names = ["None (no filter)"] + list(saved_screens.keys())
                # One-shot sync: only update widget when a screen was just saved/edited.
                # We must NOT override widget state on normal reruns — that would
                # undo the user's own dropdown selection.
                _sync_target = st.session_state.get("_sync_to_screen")
                if _sync_target:
                    if _sync_target in screen_names:
                        st.session_state["chosen_screen_select"] = _sync_target
                    del st.session_state["_sync_to_screen"]
                elif st.session_state.get("chosen_screen_select") not in screen_names:
                    # Widget value became invalid (screen deleted) — reset to no-filter
                    st.session_state["chosen_screen_select"] = "None (no filter)"
                chosen_screen = st.selectbox(
                    "Load Saved Screen",
                    options=screen_names,
                    key="chosen_screen_select",
                )
                # Always keep active_screen and conditions in sync with the dropdown
                st.session_state["active_screen"] = chosen_screen
                if chosen_screen != "None (no filter)" and chosen_screen in saved_screens:
                    screen_def = saved_screens[chosen_screen]
                    st.session_state["active_conds"] = screen_def.get("conditions", [])
                    st.session_state["active_conns"] = screen_def.get("connectors", [])
                else:
                    st.session_state["active_conds"] = []
                    st.session_state["active_conns"] = []

                # Edit / Delete buttons for the selected screen
                if chosen_screen != "None (no filter)":
                    _col_edit, _col_del = st.columns([1, 1])
                    with _col_edit:
                        if st.button("Edit Screen", key="edit_screen", use_container_width=True):
                            import copy as _copy
                            _sdef = saved_screens[chosen_screen]
                            st.session_state["show_screen_builder"] = True
                            st.session_state["sb_edit_mode"] = True
                            st.session_state["sb_original_name"] = chosen_screen
                            st.session_state["sb_screen_name"] = chosen_screen
                            st.session_state["sb_conditions"] = _copy.deepcopy(
                                _sdef.get("conditions", [{"field": "Action", "operator": "==", "value": "BULLISH"}])
                            )
                            st.session_state["sb_connectors"] = list(_sdef.get("connectors", []))
                            st.rerun()
                    with _col_del:
                        if st.button("Delete", key="del_screen", use_container_width=True):
                            del saved_screens[chosen_screen]
                            save_screens_to_file(saved_screens)
                            st.session_state["saved_screens"] = saved_screens
                            st.session_state["active_screen"] = "None (no filter)"
                            st.session_state["active_conds"] = []
                            st.session_state["active_conns"] = []
                            st.session_state["chosen_screen_select"] = "None (no filter)"
                            st.rerun()

            col_new, col_run = st.columns([1, 1])
            with col_new:
                if st.button("+ New Screen", use_container_width=True, key="open_screen_builder"):
                    # Reset builder state
                    st.session_state["show_screen_builder"] = True
                    st.session_state["sb_conditions"] = [
                        {"field": "Action", "operator": "==", "value": "BULLISH"}
                    ]
                    st.session_state["sb_connectors"] = []
                    st.session_state["sb_screen_name"] = ""
                    st.rerun()
            with col_run:
                screener_btn = st.button(
                    "Run Screener",
                    type="primary",
                    use_container_width=True,
                    key="run_screener_btn",
                )
        elif daily_picks_mode:
            st.markdown("**Daily Trade Picks** scans the Nifty 500 universe for today's best swing/momentum setups, factoring in global macro news.")
            picks_universe = st.selectbox(
                "Scan Universe",
                ["Nifty 500 (live, ~500 stocks)", "Nifty Midcap 150 (live)", "Nifty Smallcap 250 (live)",
                 "Curated 300 (fast)"],
            )
            max_scan = st.slider("Max stocks to scan", 50, 500, 150, step=50)
            daily_picks_btn = st.button("Run Today's Scan", type="primary", use_container_width=True)
        else:
            if ticker:
                st.markdown(f"**Ticker:** `{ticker}`")

            analyze_btn = st.button(
                "Analyze Stock",
                type="primary",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown(
            "**Disclaimer:** This tool is for **educational purposes only**. "
            "Signals shown (Bullish/Neutral/Bearish) are model outputs — "
            "**NOT buy/sell/hold recommendations**. "
            "The creator is not a SEBI/SEC-registered adviser. "
            "Data may be inaccurate or delayed. "
            "Always consult a qualified financial adviser before investing."
        )

    # Page title adapts to market
    if market == "India":
        st.title("Indian Stock Market Analysis Tool")
    else:
        st.title("US Stock Market Analysis Tool")
    st.caption(
        "Educational Tool — For Research Purposes Only | "
        "Technical Analysis + News Sentiment + Quarterly Fundamentals | "
        "NOT Investment Advice"
    )

    # Main content
    if not screener_mode and analyze_btn and ticker:
        # Fetch data
        with st.spinner("Fetching stock data..."):
            stock_data = fetch_stock_data(ticker)

        if stock_data["status"] == "error":
            st.error(f"Failed to fetch stock data: {stock_data.get('message', 'Unknown error')}")
            return

        # Derive sector dynamically from yfinance info
        sector = get_stock_sector(stock_data.get("info", {}))
        selected_stock = stock_name

        with st.sidebar:
            st.markdown(f"**Sector:** {sector}")

        # Fetch news and OI in parallel (both are network-bound)
        symbol_base = ticker.replace(".NS", "") if market == "India" else None
        with st.spinner("Fetching news & market data..."):
            with ThreadPoolExecutor(max_workers=2) as pool:
                news_future = pool.submit(fetch_news_headlines, selected_stock, sector, market)
                oi_future = pool.submit(fetch_oi_data, symbol_base) if symbol_base else None
            news_data = news_future.result()
            oi_result = oi_future.result() if oi_future else {"status": "error"}

        # Run all local analyses (CPU-bound, no network)
        with st.spinner("Running analysis..."):
            technical_result = calculate_technical_indicators(stock_data["history"])
            sentiment_result = analyze_sentiment(news_data)
            fundamental_result = analyze_fundamentals(
                stock_data.get("quarterly_income"),
                stock_data.get("info", {}),
                balance_sheet_df=stock_data.get("quarterly_balance"),
            )
            key_metrics = compute_key_metrics(
                stock_data.get("info", {}),
                stock_data.get("annual_income"),
                stock_data.get("annual_balance"),
                stock_data.get("cashflow"),
            )

        # Generate prediction
        prediction = generate_prediction(technical_result, sentiment_result, fundamental_result, oi_data=oi_result)

        if prediction["status"] == "error":
            st.error(prediction.get("message", "Could not generate prediction."))
            return

        # Display results
        action = prediction["action"]
        confidence = prediction["confidence"]
        score = prediction["score"]

        # Calculate trade levels
        trade_levels = None
        if technical_result.get("status") == "ok" and action != "NEUTRAL":
            trade_levels = calculate_trade_levels(technical_result, action)

        # Trade recommendation header
        st.header(f"Signal for {selected_stock}")
        render_trade_recommendation(prediction, trade_levels, oi_result, technical_result)

        # Confidence gauge + component weights
        col_gauge, col_weights = st.columns([1, 1])
        with col_gauge:
            gauge_fig = render_gauge_chart(confidence, action)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_weights:
            st.subheader("Component Weights")
            weights = prediction.get("weights", {})
            components = prediction.get("components", {})
            comp_labels = {
                "technical": "Technical", "fundamental": "Fundamental",
                "sentiment": "Sentiment", "open_interest": "Open Interest (OI)",
            }
            for comp_name, weight in weights.items():
                comp_score = components.get(comp_name, 0)
                label = comp_labels.get(comp_name, comp_name.capitalize())
                color = "#00c853" if comp_score > 0.1 else "#ff1744" if comp_score < -0.1 else "#ffc107"
                st.markdown(
                    f"**{label}** (weight: {weight*100:.0f}%) — "
                    f"Score: <span style='color:{color}'>{comp_score:+.3f}</span>",
                    unsafe_allow_html=True,
                )
            st.markdown("---")
            st.markdown(f"**Final Weighted Score:** {score:+.3f}")
            st.markdown(f"**Thresholds:** BULLISH >= +{BUY_THRESHOLD} | BEARISH <= {SELL_THRESHOLD}")

        st.markdown("---")

        # Analysis Breakdown
        st.header("Analysis Breakdown")
        render_key_metrics(key_metrics)
        st.markdown("---")
        render_component_breakdown(technical_result, sentiment_result, fundamental_result, prediction)

        st.markdown("---")

        # Price chart
        st.header("Price Chart")
        price_fig = render_price_chart(stock_data["history"], selected_stock, market=market)
        st.plotly_chart(price_fig, use_container_width=True)

        st.markdown("---")

        # News Headlines
        st.header("News Headlines & Sentiment")
        render_news_table(sentiment_result)

        # Final disclaimer
        st.markdown("---")
        st.warning(
            "EDUCATIONAL TOOL — NOT FINANCIAL ADVICE: The Bullish/Neutral/Bearish signal above is "
            "generated by automated mathematical models using publicly available data. "
            "It is NOT a recommendation to buy, sell, or hold any security. "
            "The creator is not a SEBI-registered or SEC-registered investment adviser. "
            "Past performance does not guarantee future results. Data may be delayed or inaccurate. "
            "Always consult a qualified, licensed financial adviser before making investment decisions."
        )
    elif daily_picks_mode:
        st.header("Daily Trade Picks")
        st.caption("Swing & momentum setups filtered from today's market, adjusted for global macro news.")

        # Resolve universe
        # Maps display name → (nse_key, curated_fallback_dict)
        universe_map = {
            "Nifty 500 (live, ~500 stocks)": ("nifty500",    None),
            "Nifty Midcap 150 (live)":       ("midcap150",   MIDCAP_STOCKS),
            "Nifty Smallcap 250 (live)":     ("smallcap250", SMALLCAP_STOCKS),
            "Curated 300 (fast)":            (None,          None),
        }
        idx_key, curated_fallback = universe_map.get(picks_universe, (None, None))

        def _build_curated_all():
            result = {}
            for cat_dict in STOCK_CATEGORIES.values():
                for n, t in cat_dict.items():
                    if t not in result.values():
                        result[n] = t
            return result

        if daily_picks_btn:
            with st.spinner("Fetching stock universe..."):
                if idx_key:
                    universe = fetch_nse_index_constituents(idx_key)
                    if not universe:
                        # Use the matching curated dict, not the whole combined list
                        fallback = curated_fallback or _build_curated_all()
                        st.warning(
                            f"Live NSE fetch failed — using curated {picks_universe.split('(')[0].strip()} "
                            f"list ({len(fallback)} stocks) as fallback."
                        )
                        universe = fallback
                else:
                    universe = _build_curated_all()

            if not universe:
                st.error("Could not build stock universe. Try again.")
            else:
                st.info(f"Scanning {min(max_scan, len(universe))} stocks from {picks_universe}...")
                picks_results, macro_score, macro_headlines = run_daily_picks(
                    universe, market="India", max_scan=max_scan
                )
                st.session_state["daily_picks_results"] = picks_results
                st.session_state["daily_picks_macro_score"] = macro_score
                st.session_state["daily_picks_macro_headlines"] = macro_headlines

        picks_results = st.session_state.get("daily_picks_results")
        macro_score = st.session_state.get("daily_picks_macro_score", 0.0)
        macro_headlines = st.session_state.get("daily_picks_macro_headlines", [])

        if picks_results is not None:
            render_daily_picks(picks_results, macro_score, macro_headlines, market="India")
        elif not daily_picks_btn:
            st.info("Choose your scan universe above and click **Run Today's Scan** to see today's trade setups.")

    elif screener_mode:
        # Show screen builder if requested
        if st.session_state.get("show_screen_builder"):
            render_screen_builder()
        else:
            if screener_btn:
                results = run_screener(screener_stocks, market=market)
                st.session_state["screener_results"] = results
                # Clear active_screen filter when re-running screener
                # (keep user's choice)

            results = st.session_state.get("screener_results")
            if results:
                # Apply screen filter — read conditions stored directly in session state
                active_screen = st.session_state.get("active_screen", "None (no filter)")
                active_conds = st.session_state.get("active_conds", [])
                active_conns = st.session_state.get("active_conns", [])
                filtered_results = results

                if active_screen and active_screen != "None (no filter)" and active_conds:
                    df_all = pd.DataFrame(results)
                    df_filtered = apply_screen_filters(df_all, active_conds, active_conns)
                    filtered_results = df_filtered.to_dict("records")
                    # Build a human-readable summary of the active conditions
                    cond_parts = []
                    for i, cond in enumerate(active_conds):
                        if i > 0 and i - 1 < len(active_conns):
                            cond_parts.append(active_conns[i - 1])
                        cond_parts.append(
                            f"{cond.get('field')} {cond.get('operator')} {cond.get('value')}"
                        )
                    cond_summary = "  |  ".join(cond_parts)
                    st.header("Stock Screener Results")
                    st.markdown(
                        f"**Active Screen:** `{active_screen}` | "
                        f"Filter: _{cond_summary}_ | "
                        f"Showing **{len(filtered_results)}** of {len(results)} stocks"
                    )
                else:
                    st.header("Stock Screener Results")

                if filtered_results:
                    render_screener_results(filtered_results)
                else:
                    st.warning("No stocks matched the active screen filter. Try adjusting the conditions.")
            elif screener_btn:
                st.warning("No stocks could be analyzed. Please try again.")
            else:
                _idle_screens = st.session_state.get("saved_screens", {})
                if not _idle_screens:
                    _idle_screens = load_saved_screens()
                if _idle_screens:
                    st.subheader("Saved Screens")
                    _active = st.session_state.get("active_screen", "None (no filter)")
                    for _sname, _sdef in _idle_screens.items():
                        _is_active = _sname == _active
                        with st.expander(
                            ("Active  " if _is_active else "") + _sname,
                            expanded=_is_active,
                        ):
                            _conds = _sdef.get("conditions", [])
                            _conns = _sdef.get("connectors", [])
                            if _conds:
                                _parts = []
                                for _ci, _c in enumerate(_conds):
                                    if _ci > 0 and _ci - 1 < len(_conns):
                                        _parts.append(f"**{_conns[_ci - 1]}**")
                                    _parts.append(
                                        f"`{_c.get('field')} {_c.get('operator')} {_c.get('value')}`"
                                    )
                                st.markdown("  ".join(_parts))
                            else:
                                st.caption("No conditions defined.")
                    st.info("Select a scope above and click **Run Screener** to apply the active filter.")
                else:
                    st.info("Select a scope and click **Run Screener** to scan stocks. Use **+ New Screen** to build a custom filter.")
    else:
        st.info("Select a stock from the sidebar and click **Analyze Stock** to begin.")


if __name__ == "__main__":
    main()
