# Streamlit Model Artifacts

This directory is the version-controlled deployment source for model bundles used by Streamlit.

Each horizon uses one directory:

- `60m/`
- `4h/`
- `1d/`
- `2d/`
- `7d/`

A deployable horizon directory must contain:

- `manifest.json`
- the model file declared by the manifest
- the calibrator file declared by the manifest

The application verifies SHA-256 checksums, configuration identity, horizon identity, feature order, and comparison-data requirements before loading any object.

Joblib files are executable pickle content. Commit and deploy only artifacts created by this trusted repository's training workflow. Do not accept model files from users or arbitrary URLs.
