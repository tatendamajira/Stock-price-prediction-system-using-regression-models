import base64
import io
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import time
from django.contrib.auth.decorators import login_required
from django.contrib import messages

# Set up logging for debugging and error tracking
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def compute_technical_indicators(data):
    """
    Compute technical indicators:
      - EMA (50 & 200)
      - RSI (14-day)
      - ATR (14-day)
      - MACD (12, 26, 9)
      - Bollinger Bands (20-day SMA and 2 std dev)
    """
    # EMA calculations
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # RSI calculation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # ATR calculation
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    data['ATR'] = np.maximum(np.maximum(high_low, high_close), low_close).rolling(window=14).mean()

    # MACD Calculation
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands Calculation
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std().squeeze()  # Ensure Series
    data['Bollinger_Upper'] = data['SMA_20'] + (std * 2)
    data['Bollinger_Lower'] = data['SMA_20'] - (std * 2)

    return data


def plot_historical_chart(data, stock_symbol):
    """
    Generate a historical chart with Close, EMA, and Bollinger Bands.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, data['EMA_50'], label='50-Period EMA', color='red', linestyle='--')
    plt.plot(data.index, data['EMA_200'], label='200-Period EMA', color='green', linestyle='--')
    plt.fill_between(data.index, data['Bollinger_Lower'], data['Bollinger_Upper'],
                     color='gray', alpha=0.3, label='Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Historical Prices for {stock_symbol}')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    historical_image = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    plt.close()
    return historical_image

def plot_prediction_chart(df, X_test, y_test, predictions, future_days, future_predictions, stock_symbol, model_choice):
    """
    Generate a prediction chart displaying historical data, model predictions,
    and forecasted prices. Annotate the chart with the predicted price movement pattern.
    """
    plt.figure(figsize=(14, 8))
    plt.plot(df['Day'], df['Close'], label='Historical Close', color='blue')
    plt.scatter(X_test, y_test, label='Test Data', color='green', alpha=0.6)
    plt.scatter(X_test, predictions, label='Model Predictions', color='red', alpha=0.6)
    plt.plot(future_days, future_predictions, label='Future Predictions', color='orange', linestyle='--')
    
    # Determine predicted price movement pattern from forecasted values
    if future_predictions[-1] > future_predictions[0]:
        pattern = "Uptrend"
    elif future_predictions[-1] < future_predictions[0]:
        pattern = "Downtrend"
    else:
        pattern = "Sideways"
        
    # Annotate the chart with the predicted movement pattern
    plt.annotate(f"Predicted Pattern: {pattern}",
                 xy=(future_days[len(future_days)//2], future_predictions[len(future_predictions)//2]),
                 xytext=(0.5, 0.9), textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, backgroundcolor='white')
    
    plt.xlabel('Day Index')
    plt.ylabel('Close Price')
    plt.title(f'Stock Price Forecast for {stock_symbol} using {model_choice.title()} Model')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    prediction_image = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    plt.close()
    return prediction_image

def fetch_stock_data(request):
    """
    Fetch historical stock data from Yahoo Finance, perform regression,
    predict future prices, and align the trading strategy with the predicted trend.
    Supports multiple regression models:
      - Linear Regression
      - Polynomial Regression (non-linear)
      - Random Forest Regression
    """
    if request.method == 'POST':
        stock_symbol = request.POST.get('symbol', '').upper()
        period = request.POST.get('period', '1y')
        interval = request.POST.get('interval', '1d')
        model_choice = request.POST.get('model_choice', 'linear')

        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                data = yf.download(stock_symbol, period=period, interval=interval)
                if data.empty:
                    logger.error(f"No data found for {stock_symbol}")
                    return render(request, 'fetch.html', {'error': f'No data found for symbol {stock_symbol}'})
                break  # Exit loop if successful
            except yf.YFRateLimitError:
                logger.warning(f"Rate limit exceeded for {stock_symbol}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            except Exception as e:
                logger.exception("Error fetching stock data")
                return render(request, 'fetch.html', {'error': f'Error fetching data: {str(e)}'})
        else:
            # If all retries fail
            logger.error(f"Failed to fetch data for {stock_symbol} after {max_retries} attempts.")
            return render(request, 'fetch.html', {'error': 'Rate limit exceeded. Please try again later.'})

        # Compute technical indicators
        data = compute_technical_indicators(data)

        # Generate historical chart
        historical_image_base64 = plot_historical_chart(data, stock_symbol)

        # Prepare data for regression model using day index as feature
        df = data[['Close', 'High', 'Low']].copy().dropna()
        df['Day'] = np.arange(len(df))
        X = df[['Day']].values
        y = df['Close'].values

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Choose regression model
        if model_choice == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_choice == 'polynomial':
            model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Evaluate model performance
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Forecast future stock prices for next 30 days
        future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
        future_predictions = model.predict(future_days)

        # Determine predicted price movement pattern from forecasted values
        if future_predictions[-1] > future_predictions[0]:
            predicted_pattern = "Uptrend"
        elif future_predictions[-1] < future_predictions[0]:
            predicted_pattern = "Downtrend"
        else:
            predicted_pattern = "Sideways"

        # Get current technical indicator values
        current_close = df['Close'].iloc[-1]
        atr_value = data['ATR'].iloc[-1]
        current_rsi = data['RSI'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_Signal'].iloc[-1]

        # Align trading strategy with the predicted pattern
        if predicted_pattern == "Uptrend":
            recommended_action = "Buy"
            recommended_entry = current_close  # Using current close as entry
            stop_loss_price = current_close - atr_value
            take_profit_price = current_close + 2 * atr_value
        elif predicted_pattern == "Downtrend":
            recommended_action = "Sell"
            recommended_entry = current_close
            stop_loss_price = current_close + atr_value
            take_profit_price = current_close - 2 * atr_value
        else:
            recommended_action = "Hold"
            if isinstance(future_predictions, np.ndarray):
                recommended_entry = float(future_predictions[0])
            else:
                recommended_entry = float(future_predictions.iloc[0])
            stop_loss_percentage = 3
            take_profit_percentage = 5
            stop_loss_price = recommended_entry * (1 - stop_loss_percentage / 100)
            take_profit_price = recommended_entry * (1 + take_profit_percentage / 100)

        risk_management = {
            'action': recommended_action,
            'predicted_pattern': predicted_pattern,
            'entry_price': recommended_entry,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'atr_value': atr_value,
            'current_rsi': current_rsi,
            'macd': macd,
            'macd_signal': macd_signal
        }

        # Generate prediction chart with annotated predicted price movement pattern
        prediction_image_base64 = plot_prediction_chart(
            df, X_test, y_test, predictions, future_days, future_predictions,
            stock_symbol, model_choice
        )

        # Prepare context for the results template
        context = {
            'symbol': stock_symbol,
            'historical_image_base64': historical_image_base64,
            'prediction_image_base64': prediction_image_base64,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model_choice': model_choice.title(),
            'recommended_entry': recommended_entry,
            'trading_action': recommended_action,
            'current_close': current_close,
            'risk_management': risk_management
        }
        return render(request, 'result.html', context)

    # GET request: display the input form
    return render(request, 'fetch.html')


# Accounts Views
def login_view(request):
    """
    Handle user login functionality.
    """
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {username}!")
                return redirect('fetch_stock_data')
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})


def register_view(request):
    """
    Handle user registration functionality.
    """
    from .forms import RegisterForm  # Import here to avoid circular imports
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f"Account created successfully! Welcome, {user.username}!")
            return redirect('fetch_stock_data')
        else:
            messages.error(request, "Registration failed. Please correct the errors below.")
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})


def logout_view(request):
    """
    Handle user logout and redirect to the login page.
    """
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect('login')

@login_required(login_url='login')
def list_symbols(request):
    """
    Provide a list of stock symbols for display.
    """
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "BRK.B", "V",
        "JPM", "JNJ", "WMT", "PG", "DIS", "HD", "PYPL", "MA", "VZ", "PFE",
        "KO", "PEP", "MRNA", "CSCO", "ADBE", "CMCSA", "INTC", "XOM", "NKE", "ABBV",
        "BABA", "ORCL", "IBM", "AMD", "T", "GS", "CAT", "GE", "MCD", "BA",
        "CRM", "HON", "UNH", "TMO", "LOW", "LMT", "SBUX", "TXN", "CVX", "COST",
        "SPY", "QQQ", "DIA", "IWM", "ARKK", "F", "GM", "UBER", "LYFT", "PLTR",
        "SQ", "PYPL", "SHOP", "SNAP", "TWTR", "ZM", "ROKU", "BIDU", "JD", "TCEHY",
        "TSM", "ASML", "QCOM", "TXN", "INTC", "MU", "MRVL", "LRCX", "AMAT", "KLAC",
        "BMY", "GILD", "VRTX", "REGN", "ILMN", "ISRG", "BIIB", "ALGN", "ZBH", "EW",
        "KO", "PEP", "MDLZ", "KHC", "CPB", "GIS", "SJM", "HSY", "HRL", "MKC",
        "TGT", "WBA", "KR", "CVS", "DG", "DLTR", "BJ", "O", "SPG", "VICI",
        "MS", "GS", "C", "BAC", "WFC", "USB", "PNC", "TFC", "BK", "SCHW",
        "CSX", "NSC", "UNP", "KSU", "LUV", "AAL", "DAL", "UAL", "BA", "NOC",
        "RTX", "LMT", "GD", "TXT", "HII", "LDOS", "HON", "EMR", "ETN", "ROK",
        "FCX", "NUE", "STLD", "X", "CLF", "MT", "VALE", "RIO", "BHP", "GLNCY",
        "FDX", "UPS", "EXPD", "CHRW", "XPO", "JBHT", "KNX", "LSTR", "SAIA", "WERN",
        "DE", "AGCO", "CNHI", "KUBTY", "FMC", "MOS", "NTR", "CF", "BG", "ADM",
        "ZTS", "IDXX", "ABT", "DHR", "BAX", "BDX", "SYK", "EW", "HOLX", "VAR",
        "T", "TMUS", "VZ", "CHTR", "CMCSA", "DISH", "LSXMK", "LBRDA", "SIRI", "ATUS",
        "GOOGL", "MSFT", "AAPL", "META", "AMZN", "TWLO", "NFLX", "SNOW", "U", "DOCU",
        "PANW", "CRWD", "ZS", "FTNT", "OKTA", "TENB", "RPD", "CYBR", "QLYS", "SPLK",
        "IBM", "ACN", "ORCL", "SAP", "NOW", "INTU", "CDNS", "ADBE", "PTC", "ANSS",
        "TSM", "ASML", "AMD", "NVDA", "QCOM", "TXN", "INTC", "AVGO", "MRVL", "ADI",
        "MU", "LRCX", "AMAT", "KLAC", "ONTO", "UCTT", "ICHR", "COHU", "ACMR", "ASYS",
        "MRNA", "PFE", "BNTX", "NVAX", "AZN", "GILD", "LLY", "REGN", "VRTX", "BIIB",
        "BMY", "ABBV", "GSK", "JNJ", "MRK", "NVO", "SNY", "RHHBY", "TAK", "VTRS",
        "COST", "WMT", "TGT", "BJ", "KR", "DLTR", "DG", "WBA", "CVS", "RAD",
        "MCD", "SBUX", "DPZ", "YUM", "CMG", "WEN", "QSR", "SHAK", "EAT", "CAKE",
        "PEP", "KO", "MNST", "KDP", "PRMW", "CELH", "FIZZ", "BF.B", "STZ", "SAM",
        "XOM", "CVX", "COP", "EOG", "PXD", "MPC", "VLO", "PSX", "HES", "DVN",
        "OKE", "WMB", "ET", "TRGP", "KMI", "EPD", "ENB", "PAA", "MPLX", "CEQP",
        "LUV", "DAL", "AAL", "UAL", "JBLU", "SAVE", "ALGT", "HA", "ALK", "SKYW",
        "DIS", "CMCSA", "NFLX", "VIAC", "DISCA", "FOXA", "SIRI", "LYV", "ROKU", "WWE",
        "MA", "V", "AXP", "PYPL", "SQ", "FISV", "ADP", "FIS", "GPN", "FLT",
        "MS", "GS", "JPM", "BAC", "C", "WFC", "USB", "PNC", "TFC", "RF",
        "MCO", "SPGI", "ICE", "NDAQ", "CME", "MKTX", "SEIC", "EVR", "CG", "KKR"
    ]

    return render(request, 'symbols.html', {'symbols': symbols})



def home(request):

    return render(request,'home.html')


# views.py
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Add to existing imports
MODEL_SAVE_PATH = 'trained_models/'

@login_required(login_url='login')
def train_model_view(request):
    if request.method == 'POST':
        # Get user inputs from form
        stock_symbol = request.POST.get('symbol', 'AAPL').upper()
        model_type = request.POST.get('model_type', 'random_forest')
        test_size = float(request.POST.get('test_size', 0.2))
        lookback = int(request.POST.get('lookback', 30))
        
        # Model-specific parameters
        n_estimators = int(request.POST.get('n_estimators', 100)) if model_type == 'random_forest' else None
        polynomial_degree = int(request.POST.get('polynomial_degree', 2)) if model_type == 'polynomial' else None
        svm_kernel = request.POST.get('svm_kernel', 'rbf')
        xgboost_estimators = int(request.POST.get('xgboost_estimators', 100))

        try:
            # Fetch and prepare data
            data = yf.download(stock_symbol, period='1y', interval='1d')
            data = compute_technical_indicators(data).dropna()
            
            # Create features and target
            features = data[['EMA_50', 'EMA_200', 'RSI', 'MACD', 'ATR', 'Bollinger_Upper', 'Bollinger_Lower']]
            target = data['Close']
            
            # Time-based split
            split_index = int(len(data) * (1 - test_size))
            X_train, X_test = features[:split_index], features[split_index:]
            y_train, y_test = target[:split_index], target[split_index:]

            # Initialize model
         
    
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            elif model_type == 'polynomial':
                model = make_pipeline(PolynomialFeatures(degree=polynomial_degree), LinearRegression())
            elif model_type == 'svm':
                model = SVR(kernel=svm_kernel)
            elif model_type == 'xgboost':
                model = XGBRegressor(n_estimators=xgboost_estimators)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=100)
            else:
                model = LinearRegression()

            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            # Save model
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            model_filename = f"{stock_symbol}_{model_type}_{int(time.time())}.joblib"
            joblib.dump(model, os.path.join(MODEL_SAVE_PATH, model_filename))

            # Generate training plot
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.index, y_test, label='Actual Prices')
            plt.plot(y_test.index, predictions, label='Predicted Prices')
            plt.title(f'{model_type.title()} Model Performance')
            plt.legend()
            
            # Save plot to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_image = base64.b64encode(buf.getvalue()).decode()
            buf.close()
            plt.close()

            return render(request, 'training_result.html', {
                'model_type': model_type,
                'symbol': stock_symbol,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'plot_image': plot_image,
                'model_filename': model_filename
            })

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return render(request, 'train_model.html', {'error': str(e)})

    # GET request - show training form
    return render(request, 'train_model.html')

# Add to urls.py

# Add new imports at the top
import requests
from textblob import TextBlob
from django.conf import settings
from collections import defaultdict
from dateutil.parser import parse  # Add this import at the top with others

def analyze_news_sentiment(articles):
    """Analyze sentiment of news articles using TextBlob with financial lexicon enhancement."""
    sentiments = []
    for article in articles:
        try:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            analysis = TextBlob(text)
            
            # Custom financial sentiment weighting
            sentiment = analysis.sentiment.polarity
            if any(word in text.lower() for word in ['earnings', 'profit', 'growth']):
                sentiment *= 1.2  # Amplify positive financial indicators
            if any(word in text.lower() for word in ['loss', 'decline', 'bankruptcy']):
                sentiment *= 1.3  # Amplify negative financial indicators
                
            sentiments.append({
                'date': article['publishedAt'].date() if article['publishedAt'] else None,
                'sentiment': max(-1, min(1, sentiment))  # Clamp between -1 and 1
            })
        except Exception as e:
            logger.error(f"Error analyzing article: {str(e)}")
    return sentiments



@login_required(login_url='login')
def news_based_prediction(request):
    """Separate view for pure news-based predictions"""
    if request.method == 'POST':
        stock_symbol = request.POST.get('symbol', '').upper()
        context = {'symbol': stock_symbol}
        
        try:
            # 1. Fetch financial news
            api_key = settings.NEWS_API_KEY
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': f"{stock_symbol} stock",
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': api_key,
                'pageSize': 50
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            articles = response.json().get('articles', [])

            # Convert publishedAt strings to datetime objects
            for article in articles:
                try:
                    article['publishedAt'] = parse(article['publishedAt'])
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Error parsing date for article: {str(e)}")
                    article['publishedAt'] = None  # Set invalid dates to None

            if not articles:
                raise ValueError("No recent news articles found")
            
            # 2. Analyze sentiment
            sentiments = analyze_news_sentiment(articles)
            df = pd.DataFrame(sentiments)
            
            # 3. Process sentiment data
            avg_sentiment = df['sentiment'].mean()
            sentiment_distribution = df['sentiment'].apply(
                lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral'
            ).value_counts().to_dict()
            
            # 4. Generate prediction
            prediction = "bullish" if avg_sentiment > 0.2 else "bearish" if avg_sentiment < -0.2 else "neutral"
            confidence = min(100, abs(avg_sentiment) * 100)
            
            # 5. Create visualization
            plt.figure(figsize=(10, 6))
            plt.hist(df['sentiment'], bins=20, color='skyblue', edgecolor='black')
            plt.title(f'News Sentiment Distribution for {stock_symbol}')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Number of Articles')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            sentiment_image = base64.b64encode(buf.getvalue()).decode()
            buf.close()
            plt.close()

            context.update({
                'prediction': prediction,
                'confidence': f"{confidence:.1f}%",
                'avg_sentiment': f"{avg_sentiment:.2f}",
                'sentiment_distribution': sentiment_distribution,
                'sentiment_image': sentiment_image,
                'articles': articles[:5]  # Show top 5 articles
            })

        except requests.exceptions.RequestException as e:
            logger.error(f"News API error: {str(e)}")
            context['error'] = "Failed to fetch news data"
        except Exception as e:
            logger.error(f"News prediction error: {str(e)}")
            context['error'] = str(e)

        return render(request, 'news_result.html', context)

    # GET request - show news prediction form
    return render(request, 'news_prediction.html')
def how(request):

    return render(request,'how.html')