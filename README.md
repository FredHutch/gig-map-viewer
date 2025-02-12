# Marimo Viewer: Cirro
Visualization of data managed in the Cirro data platform

The simple [marimo](https://marimo.io) app contained in this repository
includes the code needed to:

- Load the Cirro client library
- Authenticate the user's identity
- Select the Cirro project containing the data of interest
- Select the specific dataset to load
- Select a single file from that dataset

This provides a starting place for building any type of visualization app
which loads data stored within the user's Cirro account.

## Development

Set up your development environment:

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Launch the app in editable notebook format:

```
marimo edit app.py
```

Launch the app locally via HTML-WASM

```
rm -rf test_build;
marimo export html-wasm app.py -o test_build --mode run --show-code;
python -m http.server --directory test_build;
```
