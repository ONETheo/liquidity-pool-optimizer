# Crypto Price Tracking Script

This script fetches historical price data for Bitcoin (BTC) and Sonic (S) from CoinGecko and exports the data to a Google Sheet.

## Features

- Fetches daily price data for the last 30 days
- Fetches hourly price data for the last 30 hours
- Creates a Google Sheet with separate tabs for each cryptocurrency
- Automatically shares the sheet with a specified email address
- Uses the CoinGecko API to retrieve historical price data

## Prerequisites

- Python 3.8+
- `gspread` and `oauth2client` packages
- Google Cloud Platform account with Google Sheets API enabled
- CoinGecko API key (included in `.env` file)

## Setup

### 1. Install Required Packages

```bash
pip install gspread oauth2client pandas
```

### 2. Set Up Google Sheets API Credentials

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Enable the Google Sheets API and Google Drive API
4. Create a service account
5. Download the service account credentials as JSON
6. Replace the placeholder `credentials.json` file with your downloaded credentials

### 3. Configure the Script

The script is already configured with:
- CoinGecko API key from `.env` file
- Email address to share the spreadsheet with (`your-email@example.com`)
- Configured to track Bitcoin (BTC) and Sonic (S) tokens

## Usage

Run the script with:

```bash
python src/volatility-tracking.py
```

The script will:
1. Connect to the CoinGecko API and fetch price data
2. Create a new Google Sheet
3. Add worksheets for each cryptocurrency
4. Populate the worksheets with daily and hourly price data
5. Share the spreadsheet with the specified email address
6. Output the URL of the created spreadsheet

## Customizing

To track different cryptocurrencies, modify the `COINS` constant in the script:

```python
COINS = [
    {"id": "bitcoin", "symbol": "BTC", "name": "Bitcoin"},
    {"id": "sonic-2", "symbol": "S", "name": "Sonic"}
]
```

The `id` should match the CoinGecko ID for the cryptocurrency.

## Scheduling Regular Updates

You can set up a cron job to run the script regularly:

```bash
# Run the script daily at 1:00 AM
0 1 * * * cd /path/to/lp-optimizer && python src/volatility-tracking.py >> logs/volatility-tracking.log 2>&1
```

## Troubleshooting

If you encounter issues:

1. Check that your `credentials.json` file contains valid Google API credentials
2. Verify that the CoinGecko API key in the `.env` file is valid
3. Ensure the CoinGecko IDs for cryptocurrencies are correct

## Notes

- The script creates a new Google Sheet each time it runs. To update an existing sheet instead, modify the `run` method in the `PriceDataTracker` class.
- The free tier of CoinGecko API has rate limits. The script implements basic rate limiting, but you may need to adjust it for heavy usage. 