import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="XAI Audit",layout="wide")
st.title("Adversarial XAI Stress-Tester")
st.caption("Faithfulness audit: Grad-CAM vs SHAP on chest X-rays")

df_mask = pd.read_csv("outputs/mask_results.csv")
df_ch   = pd.read_csv("outputs/clever_hans_results.csv")
df_aopc = pd.read_csv("outputs/aopc_results.csv")

# Row 1: metric cards
c1,c2,c3 = st.columns(3)
c1.metric("Spurious Grad-CAM heatmaps",
    f"{df_mask['spurious'].mean()*100:.1f}%",
    "< 4% confidence drop when masked")
c2.metric("Clever Hans rate",
    f"{df_ch['artifact_highlighted'].mean()*100:.1f}%",
    "corner artifact highlighted")
lb = df_aopc.groupby("method")["aopc"].mean()
gap = (lb["shap"] - lb["gradcam"]) / lb["gradcam"] * 100
c3.metric("SHAP vs Grad-CAM AOPC",
    f"+{gap:.0f}%","SHAP more faithful")

st.divider()

# Row 2: leaderboard + histogram side by side
col1, col2 = st.columns(2)
with col1:
    st.subheader("Faithfulness leaderboard")
    tbl = lb.reset_index().sort_values("aopc",ascending=False)
    tbl.columns = ["Method","Mean AOPC"]
    tbl["Mean AOPC"] = tbl["Mean AOPC"].round(4)
    st.dataframe(tbl, use_container_width=True)

with col2:
    st.subheader("Confidence drop distribution")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(df_mask["drop"], bins=30,
            color="#378ADD", edgecolor="white", alpha=0.8)
    ax.axvline(0.04, color="red", linestyle="--",
               label="Spurious threshold (0.04)")
    ax.set_xlabel("Confidence drop after masking")
    ax.set_ylabel("Number of images")
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()
st.subheader("Per-method AOPC distribution")
fig2, ax2 = plt.subplots(figsize=(8,3))
for method, grp in df_aopc.groupby("method"):
    ax2.hist(grp["aopc"], bins=25, alpha=0.6, label=method)
ax2.set_xlabel("AOPC score")
ax2.set_ylabel("Count")
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)