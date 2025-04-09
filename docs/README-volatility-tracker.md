# Crypto Volatility Tracker

A tool to track price volatility of Bitcoin (BTC) and Sonic (S) tokens, collecting daily and hourly data and exporting to Google Sheets.

## Features

- Collects daily price data for the past 30 days
- Collects hourly price data for the past 30 hours
- Generates CSV files with the data
- Uploads data to Google Sheets (with proper credentials)
- Uses CoinGecko API with fallback to mock data generation
- Can be scheduled to run automatically

## Files

- `src/volatility_tracker.py` - Main entry point, combines data collection and Google Sheets updating
- `src/basic_volatility_tracker.py` - Standalone data collection script (outputs to CSV)
- `src/update_google_sheets.py` - Standalone Google Sheets update script
- `data/volatility/` - Directory containing generated CSV files

## Installation

1. Ensure you have Python 3.8+ installed
2. Install dependencies:

```bash
pip install gspread oauth2client pandas python-dotenv
```

3. Set up your Google Sheets API credentials (see below)
4. Set up your CoinGecko API key in the `.env` file (optional but recommended)

## Google Sheets API Setup

To update Google Sheets directly, you need to set up API credentials:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable the Google Sheets API and Google Drive API
4. Create a service account and download the credentials as JSON
5. Save the credentials file as `credentials.json`

The script will automatically share the created spreadsheet with `your-email@example.com`.

## Usage

### Combined Script

Run the entire process (data collection + Google Sheets update):

```bash
python src/volatility_tracker.py
```

To only collect data without updating Google Sheets:

```bash
python src/volatility_tracker.py --data-only
```

To only update Google Sheets using existing data:

```bash
python src/volatility_tracker.py --sheets-only
```

### Individual Scripts

Run only the data collection (outputs to CSV):

```bash
python src/basic_volatility_tracker.py
```

Run only the Google Sheets update (using existing CSV files):

```bash
python src/update_google_sheets.py
```

## Data Format

The script generates the following CSV files:

- `BTC_daily.csv` - Daily Bitcoin prices for the past 30 days
- `BTC_hourly.csv` - Hourly Bitcoin prices for the past 30 hours
- `S_daily.csv` - Daily Sonic prices for the past 30 days
- `S_hourly.csv` - Hourly Sonic prices for the past 30 hours

Each file contains:
- Timestamp (date or datetime)
- Price in USD

## Automatic Scheduling

To run the script automatically every day, add a cron job:

```bash
# Run daily at 1:00 AM
0 1 * * * cd /path/to/lp-optimizer && python src/volatility_tracker.py
```

## Troubleshooting

### CoinGecko API Issues

- The script attempts to use the CoinGecko API but will generate mock data if API access fails
- If you have a CoinGecko API key, add it to the `.env` file: `COINGECKO_API_KEY=your_key_here`
- The free tier of CoinGecko API has rate limits, which may cause occasional failures

### Google Sheets Issues

- Ensure you have valid Google Sheets API credentials in `credentials.json`
- Check that the service account has the necessary permissions
- If you get authentication errors, re-generate the credentials

## Data Sources

- Primary data source: CoinGecko API
- Fallback: Generated mock data based on realistic price models

## Notes

- All CSV files are stored in the `data/volatility/` directory
- The Google Sheet is created with tabs for each token (BTC, S)
- Each tab contains both daily and hourly data
- The script works without API keys, but using them provides more accurate data

## License

This project is licensed under the MIT License - see the LICENSE file for details. 