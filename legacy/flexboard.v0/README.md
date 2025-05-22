# FlexBoard (prototype)

A data analytics application for analyzing MLPerf benchmark results using Streamlit. This is a **prototype** version.

## Installation

1. Create a virtual environment

   ```sh
   python3 -m venv .venv
   ```

2. Activate the virtual environment

   ```sh
   source .venv/bin/activate
   ```

3. Install the dependencies

   ```sh
   pip install -r requirements.txt
   ```

## Running

Make sure your virtual environment is activated first.

1. Pull CMX repos

   ```sh
   cmx pull repo flexaihq@cmx4mlops --branch=dev
   cmx pull repo flexaihq@cmx4assets --branch=dev
   cmx pull repo flexaihq@cmx4experiments --branch=dev
   ```

2. Start the API backend

   ```sh
   # Terminal 1 - FastAPI
   fastapi run src/flexboard/api.py
   ```

3. Start the Streamlit frontend

   ```sh
   # Terminal 2 - Streamlit
   streamlit run src/flexboard/app.py
   ```

   Note: to disable one or several pages, use the following syntax:

   ```sh
   streamlit run src/flexboard/app.py --disable-pages <page_1_url_path> <page_2_url_path>
   ```

   The URL path for each page can be found in [app.py](src/flexboard/app.py).

## Development

### Adding a page

To add a new page to the dashboard, create a new folder `_<page_index>_<page_name>` in the `src/flexboard/st_pages/` folder. This newly created folder must contain a `_page.py` script. Add other page dependencies in the same folder.  
Once you're done with this, include the page in the app: add a new `st.Page` in the Streamlit navigation in `src/flexboard/app.py`, with the title you want.
