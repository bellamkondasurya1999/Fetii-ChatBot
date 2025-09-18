# Fetii Trips Analytics - Streamlit App

A natural language interface for analyzing Fetii trip data using Streamlit and DuckDB.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push your code to a GitHub repository
2. Connect your repository to Streamlit Cloud
3. The app will automatically use the `requirements.txt` file
4. Make sure your CSV data file is in the repository root or in a `data/` folder

## Data Files

The app looks for trip data in the following locations (in order):
- `data/trips.csv` (recommended)
- `trips.csv`
- `FetiiAI_Data_Austin.xlsx - Trip Data.csv`
- `data/FetiiAI_Data_Austin.xlsx - Trip Data.csv`
- `Trip Data.csv`
- `data/Trip Data.csv`

## Features

- Natural language queries about trip data
- Interactive maps showing trip hotspots
- Time-based filtering (last month, weekend, etc.)
- Group size analysis
- Venue-specific queries (Moody Center, Rainey Street, etc.)

## Troubleshooting

If you encounter import errors on Streamlit Cloud:
1. Check that all dependencies are listed in `requirements.txt`
2. Ensure your Python version is compatible (see `runtime.txt`)
3. Verify that your data files are in the correct location
