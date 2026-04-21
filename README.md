# Adversarial XAI Stress-Tester — Radiology AI

Systematic faithfulness audit of Grad-CAM, SHAP on chest X-rays.

## Key results
| Metric | Result |
|---|---|
| Spurious Grad-CAM heatmaps | 35.9% |
| Clever Hans artifact rate | 65.94% |
| SHAP AOPC advantage | 92% |

## Stack
PyTorch · Captum · torchxrayvision · NIH CXR14

## Run
pip install -r requirements.txt
streamlit run dashboard/app.py

## Live demo
[Streamlit app](https://xai-stress-tester.streamlit.app/)
