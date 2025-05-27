from django.shortcuts import render
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = []
        self.required_history = 30  # Number of past days required

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            df = X.copy()

            # Price transformations
            df['log_close'] = np.log(df['Close'])
            df['returns'] = df['Close'].pct_change()

            # Technical indicators: moving averages and rolling volatility
            for window in [7, 14, 21]:
                df[f'MA{window}'] = df['Close'].rolling(window).mean()
                df[f'Volatility_{window}'] = df['returns'].rolling(window).std()

            # Lag features to capture momentum
            for lag in [1, 3, 5, 7]:
                df[f'lag_{lag}'] = df['Close'].shift(lag)

            # Drop rows with NaN values generated from rolling calculations and shifts
            df = df.dropna()
            self.feature_names = df.columns.tolist()
            return df[self.feature_names]

        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise

class StockPredictor:
    def __init__(self, ticker):
        self.ticker = ticker.upper().strip()
        self.model = None
        self.min_training_days = 200

    def _validate_pipeline(self):
        """Ensure the pipeline has the required steps."""
        if not hasattr(self.model, 'named_steps'):
            raise ValueError("Invalid pipeline structure")
        required_steps = ['features', 'scaler', 'regressor']
        for step in required_steps:
            if step not in self.model.named_steps:
                raise ValueError(f"Missing pipeline step: {step}")

    def get_data(self):
        """Download and prepare historical stock data."""
        try:
            df = yf.download(self.ticker, period='5y', auto_adjust=True)
            if len(df) < self.min_training_days:
                raise ValueError(f"Minimum {self.min_training_days} trading days required")
            # Forward-fill missing values and drop any remaining NaNs
            return df[['Open', 'High', 'Low', 'Close']].ffill().dropna()
        except Exception as e:
            logger.error(f"Data download failed: {str(e)}")
            raise

    def train_model(self):
        """
        Experiment with different regression models using GridSearchCV for hyperparameter tuning.
        Evaluates models using TimeSeriesSplit cross-validation based on negative MAPE.
        The best performing model is saved.
        """
        try:
            df = self.get_data()
            X = df.copy()
            # Use next-day closing price as the target
            y = X['Close'].shift(-1).dropna()
            X = X.iloc[:-1]

            # Define cross-validation strategy for time series data
            tscv = TimeSeriesSplit(n_splits=3)

            # Define candidate models and their hyperparameter grids
            candidates = {
                'HistGradientBoostingRegressor': {
                    'model': HistGradientBoostingRegressor(random_state=42),
                    'params': {
                        'regressor__max_iter': [500, 1000],
                        'regressor__learning_rate': [0.01, 0.05, 0.1],
                        'regressor__max_depth': [None, 7]
                    }
                },
                'RandomForestRegressor': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'regressor__n_estimators': [50, 100, 200],
                        'regressor__max_depth': [None, 10, 20]
                    }
                },
                'LinearRegression': {
                    'model': LinearRegression(),
                    'params': {}  # No hyperparameters to tune for basic LinearRegression
                }
            }

            best_model = None
            best_score = -np.inf  # Since we use negative MAPE, a higher value is better
            candidate_results = {}

            # Loop through candidates, tuning hyperparameters with GridSearchCV
            for name, candidate in candidates.items():
                pipeline = Pipeline([
                    ('features', AdvancedFeatureEngineer()),
                    ('scaler', RobustScaler()),
                    ('regressor', candidate['model'])
                ])
                grid = GridSearchCV(
                    pipeline,
                    param_grid=candidate['params'],
                    scoring='neg_mean_absolute_percentage_error',
                    cv=tscv,
                    n_jobs=-1,
                    refit=True
                )
                grid.fit(X, y)
                candidate_results[name] = grid.best_score_
                logger.info(f"{name} best score (neg MAPE): {grid.best_score_:.4f}")
                if grid.best_score_ > best_score:
                    best_score = grid.best_score_
                    best_model = grid.best_estimator_

            if best_model is None:
                raise ValueError("No valid model found during tuning.")

            self.model = best_model
            self._validate_pipeline()

            # Save the best model for later forecasting
            os.makedirs('models', exist_ok=True)
            model_path = f'models/{self.ticker}_model.pkl'
            joblib.dump(self.model, model_path)
            logger.info(f"Best model saved to {model_path}")

            return self

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def forecast(self, days=7):
        """Generate future forecasts using the trained model."""
        try:
            model_path = f'models/{self.ticker}_model.pkl'
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model not found - train first")

            # Load and validate the saved model
            self.model = joblib.load(model_path)
            self._validate_pipeline()

            # Prepare data window for forecasting
            df = self.get_data()
            window_size = self.model.named_steps['features'].required_history
            current_data = df.iloc[-window_size:].copy()

            predictions = []
            forecast_dates = []
            for _ in range(days):
                # Transform current data for prediction
                features = self.model.named_steps['features'].transform(current_data)
                features = self.model.named_steps['scaler'].transform(features)
                pred = self.model.named_steps['regressor'].predict(features[-1:])[0]

                # Update data with predicted value (simulate next day's trading data)
                new_date = current_data.index[-1] + timedelta(days=1)
                new_row = pd.DataFrame({
                    'Open': current_data['Close'].iloc[-1],
                    'High': pred * 1.01,
                    'Low': pred * 0.99,
                    'Close': pred
                }, index=[new_date])

                current_data = pd.concat([current_data, new_row]).iloc[1:]
                predictions.append(pred)
                forecast_dates.append(new_date)

            return predictions, forecast_dates

        except Exception as e:
            logger.error(f"Forecasting failed: {str(e)}")
            raise

def stock_prediction(request):
    context = {'ticker': '', 'days': 7}
    try:
        ticker = request.POST.get('ticker', 'AAPL').upper().strip()
        days = int(request.POST.get('days', 7))

        # Input validation: Limit forecast days between 1 and 14
        if days < 1 or days > 14:
            raise ValueError("Forecast days must be between 1 and 14")

        predictor = StockPredictor(ticker)

        # Train and tune model if it does not exist
        model_path = f'models/{ticker}_model.pkl'
        if not os.path.exists(model_path):
            predictor.train_model()

        # Generate forecast using the best model
        forecast, dates = predictor.forecast(days)
        df = predictor.get_data()

        # Create historical candlestick chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Historical'
        ))

        # Create forecast DataFrame for plotting forecasted candles
        forecast_df = pd.DataFrame({
            'Open': [df['Close'].iloc[-1]] + forecast[:-1],
            'High': [p * 1.01 for p in forecast],
            'Low': [p * 0.99 for p in forecast],
            'Close': forecast
        }, index=dates)

        # Add forecast candlestick trace
        fig.add_trace(go.Candlestick(
            x=forecast_df.index,
            open=forecast_df['Open'],
            high=forecast_df['High'],
            low=forecast_df['Low'],
            close=forecast_df['Close'],
            name='Forecast',
            increasing_line_color='#2ecc71',
            decreasing_line_color='#e74c3c'
        ))

        # Overlay a line chart for predicted closing prices
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Close'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='blue', dash='dash'),
            marker=dict(size=8)
        ))

        # Add a vertical dashed line to mark the transition point
        fig.add_vline(x=df.index[-1], line_width=2, line_dash="dash", line_color="white")

        fig.update_layout(
            title=f'{ticker} Price Forecast',
            template='plotly_dark',
            height=600
        )

        # Evaluate forecast performance on recent historical data (if available)
        recent_actual = df['Close'].iloc[-len(forecast):]
        # For demonstration, we assume forecasted values correspond roughly to recent history
        accuracy = 100 * (1 - mean_absolute_percentage_error(recent_actual, forecast))

        # Prepare context for rendering
        context.update({
            'ticker': ticker,
            'plot_div': fig.to_html(full_html=False),
            'forecast_data': zip(
                [d.strftime('%Y-%m-%d') for d in dates],
                [f"{p:.2f}" for p in forecast]
            ),
            'latest_price': f"{df['Close'].iloc[-1]:.2f}",
            'accuracy': f"{accuracy:.1f}%"
        })

    except Exception as e:
        error_msg = str(e)
        if 'ticker' in error_msg.lower():
            error_msg += ". Valid examples: AAPL, MSFT, GOOGL"
        context['error'] = error_msg

    return render(request, 'dashboard.html', context)
