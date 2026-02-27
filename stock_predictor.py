"""
Stock Market Prediction Tool (India + US)
Predicts Buy/Hold/Sell for NSE and US stocks using technical analysis,
news sentiment analysis, and quarterly fundamental analysis.
"""

# =============================================================================
# Section 1: Imports & Constants
# =============================================================================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import requests
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import io

# Technical analysis constants
SMA_SHORT = 20
SMA_LONG = 50
RSI_PERIOD = 14

# Composite weight constants
TECH_SMA_WEIGHT = 0.40
TECH_RSI_WEIGHT = 0.35
TECH_MOMENTUM_WEIGHT = 0.25

SENTIMENT_STOCK_WEIGHT = 0.50
SENTIMENT_SECTOR_WEIGHT = 0.30
SENTIMENT_MARKET_WEIGHT = 0.20

FUND_REVENUE_WEIGHT = 0.20
FUND_MARGIN_WEIGHT = 0.15
FUND_PROFIT_WEIGHT = 0.20
FUND_DEBT_EQUITY_WEIGHT = 0.20
FUND_CURRENT_RATIO_WEIGHT = 0.10
FUND_ROE_WEIGHT = 0.15

# Final prediction weights
WEIGHT_TECHNICAL = 0.45
WEIGHT_FUNDAMENTAL = 0.30
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

STOCK_CATEGORIES = {
    "Large Cap (Nifty 50)": NIFTY_50,
    "Nifty Next 50": NIFTY_NEXT_50,
    "Mid Cap": MIDCAP_STOCKS,
    "Small Cap": SMALLCAP_STOCKS,
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
        stock = yf.Ticker(ticker)
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
            feed = feedparser.parse(url)
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


# =============================================================================
# Section 5: Technical Analysis
# =============================================================================

def calculate_technical_indicators(price_df):
    """Calculate SMA crossover, RSI, and momentum signals."""
    if price_df is None or len(price_df) < SMA_LONG:
        return {"status": "insufficient_data"}

    close = price_df["Close"]

    # SMA Crossover Signal
    sma_short = close.rolling(window=SMA_SHORT).mean()
    sma_long = close.rolling(window=SMA_LONG).mean()
    latest_short = sma_short.iloc[-1]
    latest_long = sma_long.iloc[-1]

    if latest_long == 0:
        sma_signal = 0.0
    else:
        sma_diff_pct = (latest_short - latest_long) / latest_long
        sma_signal = float(np.clip(sma_diff_pct * 10, -1, 1))

    # RSI Signal
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=RSI_PERIOD).mean()

    latest_loss = loss.iloc[-1]
    if latest_loss == 0:
        rsi = 100.0
    else:
        rs = gain.iloc[-1] / latest_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    if rsi >= 70:
        rsi_signal = -((rsi - 70) / 30)  # Overbought -> sell signal
    elif rsi <= 30:
        rsi_signal = (30 - rsi) / 30  # Oversold -> buy signal
    else:
        rsi_signal = (rsi - 50) / 40  # Neutral zone, slight lean

    rsi_signal = float(np.clip(rsi_signal, -1, 1))

    # Momentum Signal (5-day vs 20-day return)
    if len(close) >= 20:
        ret_5d = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if close.iloc[-5] != 0 else 0
        ret_20d = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if close.iloc[-20] != 0 else 0
        momentum_signal = float(np.clip((ret_5d * 0.6 + ret_20d * 0.4) * 5, -1, 1))
    else:
        momentum_signal = 0.0

    # Composite technical score
    composite = (
        sma_signal * TECH_SMA_WEIGHT
        + rsi_signal * TECH_RSI_WEIGHT
        + momentum_signal * TECH_MOMENTUM_WEIGHT
    )

    return {
        "status": "ok",
        "score": float(np.clip(composite, -1, 1)),
        "sma_signal": sma_signal,
        "rsi_signal": rsi_signal,
        "rsi_value": rsi,
        "momentum_signal": momentum_signal,
        "sma_short": latest_short,
        "sma_long": latest_long,
        "current_price": float(close.iloc[-1]),
    }


# =============================================================================
# Section 6: Sentiment Analysis
# =============================================================================

def analyze_sentiment(news_dict):
    """Analyze sentiment of news headlines using VADER."""
    analyzer = SentimentIntensityAnalyzer()

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

    if available_signals == 0:
        return {"status": "insufficient_data"}

    # Compute composite with available signals
    if available_signals == 6:
        composite = (
            signals.get("revenue_growth", 0) * FUND_REVENUE_WEIGHT
            + signals.get("profit_margin", 0) * FUND_MARGIN_WEIGHT
            + signals.get("profit_growth", 0) * FUND_PROFIT_WEIGHT
            + signals.get("debt_to_equity", 0) * FUND_DEBT_EQUITY_WEIGHT
            + signals.get("current_ratio", 0) * FUND_CURRENT_RATIO_WEIGHT
            + signals.get("roe", 0) * FUND_ROE_WEIGHT
        )
    else:
        # Equal weight for available signals
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
        "available_signals": available_signals,
        "raw_margin": info.get("profitMargins"),
        "raw_de_ratio": raw_de_ratio,
        "raw_current_ratio": raw_current_ratio,
        "raw_roe": raw_roe,
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
    These do NOT affect the Buy/Hold/Sell prediction.
    """
    metrics = {
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

            results.append({
                "Stock": name,
                "Ticker": ticker,
                "Action": prediction["action"],
                "Confidence": prediction["confidence"],
                "Score": prediction["score"],
                "Tech Score": technical.get("score"),
                "Sentiment Score": sentiment.get("score"),
                "Fundamental Score": fundamental.get("score"),
                "P/E": key_metrics.get("pe_ratio"),
                "P/B": key_metrics.get("price_to_book"),
                "ROE": key_metrics.get("roe_latest"),
                "ROCE": key_metrics.get("roce_latest"),
                "Piotroski": key_metrics.get("piotroski_score"),
                "D/E": key_metrics.get("debt_to_equity"),
                "Current Ratio": key_metrics.get("current_ratio"),
                "Market Cap": key_metrics.get("market_cap"),
                "CMP": technical.get("current_price"),
            })
        except Exception:
            continue  # Skip failed stocks silently

    progress_bar.empty()
    status_text.empty()
    return results


# =============================================================================
# Section 8: Prediction Engine
# =============================================================================

def generate_prediction(technical, sentiment, fundamental):
    """Generate final Buy/Hold/Sell prediction with confidence."""
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

    # Determine action
    if final_score >= BUY_THRESHOLD:
        action = "BUY"
    elif final_score <= SELL_THRESHOLD:
        action = "SELL"
    else:
        action = "HOLD"

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
# Section 9: UI Helpers
# =============================================================================

def render_gauge_chart(confidence, action):
    """Render a Plotly gauge chart for confidence level."""
    color_map = {"BUY": "#00c853", "SELL": "#ff1744", "HOLD": "#ffc107"}
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
    """Render a Key Financial Metrics dashboard section."""
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
    buy_count = len(df[df["Action"] == "BUY"])
    hold_count = len(df[df["Action"] == "HOLD"])
    sell_count = len(df[df["Action"] == "SELL"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div style='background-color:#e8f5e9;padding:20px;border-radius:10px;text-align:center;'>"
            f"<h2 style='color:#00c853;margin:0;'>{buy_count}</h2>"
            f"<p style='margin:0;color:#2e7d32;'>stocks rated BUY</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div style='background-color:#fff8e1;padding:20px;border-radius:10px;text-align:center;'>"
            f"<h2 style='color:#ff8f00;margin:0;'>{hold_count}</h2>"
            f"<p style='margin:0;color:#f57f17;'>stocks rated HOLD</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div style='background-color:#ffebee;padding:20px;border-radius:10px;text-align:center;'>"
            f"<h2 style='color:#ff1744;margin:0;'>{sell_count}</h2>"
            f"<p style='margin:0;color:#c62828;'>stocks rated SELL</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Part B: Filter tabs
    filter_action = st.radio(
        "Filter by Action",
        options=["All", "BUY", "HOLD", "SELL"],
        horizontal=True,
    )

    if filter_action != "All":
        filtered_df = df[df["Action"] == filter_action].copy()
    else:
        filtered_df = df.copy()

    # Sort by confidence descending
    filtered_df = filtered_df.sort_values("Confidence", ascending=False).reset_index(drop=True)

    # Format percentage columns
    def _fmt_pct(val):
        if val is None or pd.isna(val):
            return "N/A"
        return f"{val * 100:.1f}%"

    def _fmt_mcap(val):
        if val is None or pd.isna(val):
            return "N/A"
        if val >= 1e12:
            return f"{val / 1e12:.2f}T"
        if val >= 1e9:
            return f"{val / 1e9:.2f}B"
        if val >= 1e7:
            return f"{val / 1e7:.2f}Cr"
        return f"{val:,.0f}"

    # Build display dataframe
    display_df = pd.DataFrame({
        "Stock": filtered_df["Stock"],
        "Action": filtered_df["Action"],
        "Confidence (%)": filtered_df["Confidence"],
        "Score": filtered_df["Score"].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "N/A"),
        "CMP": filtered_df["CMP"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"),
        "P/E": filtered_df["P/E"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"),
        "P/B": filtered_df["P/B"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"),
        "ROE": filtered_df["ROE"].apply(_fmt_pct),
        "ROCE": filtered_df["ROCE"].apply(_fmt_pct),
        "D/E": filtered_df["D/E"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"),
        "Piotroski": filtered_df["Piotroski"].apply(lambda x: f"{int(x)}/9" if pd.notna(x) else "N/A"),
        "Market Cap": filtered_df["Market Cap"].apply(_fmt_mcap),
    })

    # Part C: Data table
    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(len(display_df) * 38 + 40, 800),
        column_config={
            "Confidence (%)": st.column_config.ProgressColumn(
                "Confidence (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )

    st.caption(f"Showing {len(display_df)} of {len(df)} stocks")


def render_component_breakdown(technical, sentiment, fundamental, prediction):
    """Render 3-column breakdown of each analysis component."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Technical Analysis")
        if technical.get("status") == "ok":
            score = technical["score"]
            color = "#00c853" if score > 0.1 else "#ff1744" if score < -0.1 else "#ffc107"
            st.markdown(f"**Score:** <span style='color:{color}'>{score:+.3f}</span>", unsafe_allow_html=True)
            st.markdown(f"- RSI: {technical['rsi_value']:.1f}")
            st.markdown(f"- SMA{SMA_SHORT}: {technical['sma_short']:.2f}")
            st.markdown(f"- SMA{SMA_LONG}: {technical['sma_long']:.2f}")
            st.markdown(f"- Price: {technical['current_price']:.2f}")

            sma_label = "Bullish" if technical["sma_signal"] > 0 else "Bearish"
            rsi_label = "Overbought" if technical["rsi_value"] > 70 else "Oversold" if technical["rsi_value"] < 30 else "Neutral"
            mom_label = "Positive" if technical["momentum_signal"] > 0.1 else "Negative" if technical["momentum_signal"] < -0.1 else "Flat"
            st.markdown(f"- SMA Cross: **{sma_label}**")
            st.markdown(f"- RSI Zone: **{rsi_label}**")
            st.markdown(f"- Momentum: **{mom_label}**")
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

            st.markdown(f"- Signals Available: **{fundamental['available_signals']}/6**")
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
    """Render stock price chart with SMAs."""
    close = history["Close"]
    sma_short = close.rolling(window=SMA_SHORT).mean()
    sma_long = close.rolling(window=SMA_LONG).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history.index, y=close,
        name="Close Price", line=dict(color="#1976d2", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=history.index, y=sma_short,
        name=f"SMA {SMA_SHORT}", line=dict(color="#ff9800", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=history.index, y=sma_long,
        name=f"SMA {SMA_LONG}", line=dict(color="#e91e63", width=1, dash="dash"),
    ))
    fig.update_layout(
        title=f"{stock_name} - 1 Year Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)" if market == "US" else "Price (INR)",
        height=400,
        margin=dict(t=40, b=40, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# =============================================================================
# Section 10: Main App
# =============================================================================

def main():
    st.set_page_config(
        page_title="Stock Market Predictor",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

    # Sidebar
    with st.sidebar:
        market = st.radio("Market", ["India", "US"], horizontal=True)

        if market == "India":
            st.header("Indian Stock Market")
            input_mode = st.radio(
                "How do you want to select a stock?",
                options=["Browse by Category", "Browse by Sector", "Enter Custom Ticker", "Stock Screener"],
                horizontal=True,
            )

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
                st.markdown("Enter any NSE stock symbol (e.g. `ZOMATO`, `IRCTC`, `PAYTM`)")
                custom_ticker = st.text_input(
                    "NSE Symbol",
                    value="",
                    placeholder="e.g. ZOMATO",
                ).strip().upper()
                if custom_ticker:
                    ticker = custom_ticker if custom_ticker.endswith(".NS") else f"{custom_ticker}.NS"
                    stock_name = custom_ticker.replace(".NS", "")
                else:
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
                    st.warning(f"Could not fetch {index_name} constituents. Try again later.")
                    ticker = None
                    stock_name = None
                else:
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
                st.markdown("Enter any US stock ticker (e.g. `AAPL`, `MSFT`, `TSLA`)")
                custom_ticker = st.text_input(
                    "US Ticker",
                    value="",
                    placeholder="e.g. AAPL",
                ).strip().upper()
                if custom_ticker:
                    ticker = custom_ticker
                    stock_name = custom_ticker
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
        screener_stocks = {}
        screener_btn = False
        analyze_btn = False

        if screener_mode:
            if market == "India":
                scope_options = (
                    ["All NSE Stocks", "Curated Stocks (277)"]
                    + list(STOCK_CATEGORIES.keys())
                    + sorted(STOCK_SECTORS.keys())
                )
                screener_scope = st.selectbox("Screener Scope", options=scope_options)

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
                elif screener_scope == "Curated Stocks (277)":
                    all_stocks = {}
                    for cat_dict in STOCK_CATEGORIES.values():
                        for name, tick in cat_dict.items():
                            if tick not in all_stocks.values():
                                all_stocks[name] = tick
                    screener_stocks = all_stocks
                elif screener_scope in STOCK_CATEGORIES:
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
            screener_btn = st.button(
                "Run Screener",
                type="primary",
                use_container_width=True,
            )
        else:
            if ticker:
                st.markdown(f"**Ticker:** `{ticker}`")

            analyze_btn = st.button(
                "Analyze Stock",
                type="primary",
                use_container_width=True,
                disabled=(ticker is None),
            )

        st.markdown("---")
        st.markdown(
            "**Disclaimer:** This tool is for educational purposes only. "
            "Do not make investment decisions based solely on this analysis. "
            "Always consult a qualified financial advisor."
        )

    # Page title adapts to market
    if market == "India":
        st.title("Indian Stock Market Prediction Tool")
    else:
        st.title("US Stock Market Prediction Tool")
    st.caption("Technical Analysis + News Sentiment + Quarterly Fundamentals")

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

        with st.spinner("Fetching news headlines..."):
            news_data = fetch_news_headlines(selected_stock, sector, market=market)

        # Run analyses
        with st.spinner("Running technical analysis..."):
            technical_result = calculate_technical_indicators(stock_data["history"])

        with st.spinner("Running sentiment analysis..."):
            sentiment_result = analyze_sentiment(news_data)

        with st.spinner("Running fundamental analysis..."):
            fundamental_result = analyze_fundamentals(
                stock_data.get("quarterly_income"),
                stock_data.get("info", {}),
                balance_sheet_df=stock_data.get("quarterly_balance"),
            )

        with st.spinner("Computing key metrics..."):
            key_metrics = compute_key_metrics(
                stock_data.get("info", {}),
                stock_data.get("annual_income"),
                stock_data.get("annual_balance"),
                stock_data.get("cashflow"),
            )

        # Generate prediction
        prediction = generate_prediction(technical_result, sentiment_result, fundamental_result)

        if prediction["status"] == "error":
            st.error(prediction.get("message", "Could not generate prediction."))
            return

        # Display results
        action = prediction["action"]
        confidence = prediction["confidence"]
        score = prediction["score"]

        # Big colored action label
        action_colors = {"BUY": "#00c853", "SELL": "#ff1744", "HOLD": "#ffc107"}
        action_text_colors = {"BUY": "white", "SELL": "white", "HOLD": "black"}
        bg = action_colors.get(action, "#ffc107")
        fg = action_text_colors.get(action, "black")

        st.markdown(
            f"""
            <div style="
                background-color: {bg};
                color: {fg};
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 20px;
            ">
                <h1 style="margin:0; font-size: 3em; color: {fg};">{action}</h1>
                <p style="margin:5px 0 0 0; font-size: 1.2em; color: {fg};">
                    {selected_stock} | Score: {score:+.3f} | Confidence: {confidence:.1f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence gauge
        col_gauge, col_weights = st.columns([1, 1])
        with col_gauge:
            gauge_fig = render_gauge_chart(confidence, action)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_weights:
            st.subheader("Component Weights")
            weights = prediction.get("weights", {})
            components = prediction.get("components", {})
            for comp_name, weight in weights.items():
                comp_score = components.get(comp_name, 0)
                label = comp_name.capitalize()
                color = "#00c853" if comp_score > 0.1 else "#ff1744" if comp_score < -0.1 else "#ffc107"
                st.markdown(
                    f"**{label}** (weight: {weight*100:.0f}%) — "
                    f"Score: <span style='color:{color}'>{comp_score:+.3f}</span>",
                    unsafe_allow_html=True,
                )
            st.markdown("---")
            st.markdown(f"**Final Weighted Score:** {score:+.3f}")
            st.markdown(f"**Thresholds:** BUY >= +{BUY_THRESHOLD} | SELL <= {SELL_THRESHOLD}")

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
            "This prediction is generated using automated technical and sentiment analysis. "
            "It should NOT be used as the sole basis for investment decisions. "
            "Past performance does not guarantee future results. Always do your own research."
        )
    elif screener_mode:
        if screener_btn:
            results = run_screener(screener_stocks, market=market)
            st.session_state["screener_results"] = results

        results = st.session_state.get("screener_results")
        if results:
            st.header("Stock Screener Results")
            render_screener_results(results)
        elif screener_btn:
            st.warning("No stocks could be analyzed. Please try again.")
        else:
            st.info("Select a scope and click **Run Screener** to scan stocks.")
    else:
        st.info("Select a stock from the sidebar and click **Analyze Stock** to begin.")


if __name__ == "__main__":
    main()
