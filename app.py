# app.py
import os, re, math, joblib, io
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import streamlit as st

# ------------------------- 1. 模型定义 -------------------------
class MTWAE(nn.Module):
    def __init__(self, in_features: int, latent_size: int = 8):
        super().__init__()
        # 编码器
        self.encoder_layer = nn.Sequential(
            nn.Linear(in_features, 90), nn.LayerNorm(90), nn.LeakyReLU(),
            nn.Linear(90, 48), nn.LayerNorm(48), nn.LeakyReLU(),
            nn.Linear(48, 30), nn.LayerNorm(30), nn.LeakyReLU(),
            nn.Linear(30, latent_size),
        )
        # 解码器（这里只是占位，推断时其实用不到）
        self.decoder_layer = nn.Sequential(
            nn.Linear(latent_size, 30), nn.LayerNorm(30), nn.LeakyReLU(),
            nn.Linear(30, 48), nn.LayerNorm(48), nn.LeakyReLU(),
            nn.Linear(48, 90), nn.LayerNorm(90), nn.LeakyReLU(),
            nn.Linear(90, in_features),
        )
        # 三个性质预测头
        self.Bs_head = self._make_head(latent_size)
        self.Hc_head = self._make_head(latent_size)
        self.Dc_head = self._make_head(latent_size)

    @staticmethod
    def _make_head(latent_size):
        return nn.Sequential(
            nn.Linear(latent_size, 90), nn.LayerNorm(90), nn.LeakyReLU(),
            nn.Linear(90, 90), nn.LayerNorm(90), nn.LeakyReLU(),
            nn.Linear(90, 90), nn.LayerNorm(90), nn.LeakyReLU(),
            nn.Linear(90, 1),
        )

    # -------- 推断路径 --------
    def forward(self, X):
        z = self.encoder_layer(X)
        pred_Bs = self.Bs_head(z)
        pred_Hc = self.Hc_head(z)
        pred_Dc = self.Dc_head(z)
        return pred_Bs, pred_Hc, pred_Dc

# ----------------------- 2. 工具函数 --------------------------
PERIODIC_TABLE = [
    'Fe','B','Si','P','C','Co','Nb','Ni','Mo','Zr','Ga','Al',
    'Dy','Cu','Cr','Y','Nd','Hf','Ti','Tb','Ho','Ta','Er','Sn',
    'W','Tm','Gd','Sm','V','Pr'
]

COMP_PATTERN = re.compile(r"([A-Z][a-z]?)([\d\.]+)")

def parse_composition(comp_str: str) -> Tuple[List[str], List[float]]:
    """从 'Fe68.2Co17.5B13' 解析元素与含量"""
    matches = COMP_PATTERN.findall(comp_str.strip())
    if not matches:
        raise ValueError(f"无法解析合金成分: {comp_str}")
    elements, fracs = zip(*matches)
    return list(elements), [float(x) for x in fracs]

def to_feature_vector(elements: List[str], fracs: List[float]) -> np.ndarray:
    """将元素列表映射到 30 维特征"""
    vec = np.zeros(len(PERIODIC_TABLE), dtype=np.float32)
    for e, f in zip(elements, fracs):
        if e not in PERIODIC_TABLE:
            raise ValueError(f"元素 {e} 不在 30 元素表中")
        vec[PERIODIC_TABLE.index(e)] = f * 0.01  # 直接百分含量 → 0-1
    return vec

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """一次性加载模型与 Scaler"""
    device = torch.device("cpu")  # 推断阶段 CPU 就够用了
    model = MTWAE(in_features=len(PERIODIC_TABLE), latent_size=8).to(device)
    ckpt = torch.load("Multi_task_generative_Model_train_latent_8_sigma_8_joint_test_o1_Normalized_epoch_800.pth", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    Bs_scaler = joblib.load("Bs_scaler.pkl")
    Hc_scaler = joblib.load("Hc_scaler.pkl")
    Dc_scaler = joblib.load("Dc_scaler.pkl")
    return model, Bs_scaler, Hc_scaler, Dc_scaler, device

def predict_batch(comp_list: List[str]):
    model, Bs_s, Hc_s, Dc_s, device = load_artifacts()
    feature_mat = []
    valid_names = []
    errors = []

    for comp in comp_list:
        if not comp.strip():                # 跳过空行
            continue
        try:
            eles, fracs = parse_composition(comp)
            feature_mat.append(to_feature_vector(eles, fracs))
            valid_names.append(comp)
        except ValueError as e:
            errors.append(str(e))

    if not feature_mat:
        return pd.DataFrame(), errors

    X = torch.from_numpy(np.vstack(feature_mat)).float().to(device)

    with torch.no_grad():
        pred_Bs, pred_Hc, pred_Dc = model(X)

    # inverse_transform
    Bs_vals = Bs_s.inverse_transform(pred_Bs.cpu().numpy())
    lnHc_vals = Hc_s.inverse_transform(pred_Hc.cpu().numpy())
    Dc_vals = Dc_s.inverse_transform(pred_Dc.cpu().numpy())

    # exp 还原实际 Hc
    Hc_vals = np.exp(lnHc_vals)

    results = pd.DataFrame({
        "Composition": valid_names,
        "Bs / T": Bs_vals.flatten(),
        "Hc (A/m)": Hc_vals.flatten(),
        "ln(Hc)": lnHc_vals.flatten(),
        "Dc / mm": Dc_vals.flatten(),
    })
    return results, errors

# -------------------- 3. Streamlit 前端 -----------------------
st.set_page_config(page_title="Fe-based MGs Property Predictor",
                   page_icon="🧲", layout="centered")

st.title("🧲 Fe-based Metallic Glass Predictor (MTWAE)")

st.markdown("""
输入 **Fe-基合金** 质量百分数配比，例如：  
Fe68.2Co17.5B13Si0.5Cu0.8
Fe79.7Co6B13Si0.5Cu0.8
- 支持一次粘贴多行，<kbd>Ctrl+Enter</kbd> 或点击 **Predict** 预测。  
- Hc 显示两列：对数值 `ln(Hc)` 及还原后的实际 `Hc(A/m)`。  
""")

comp_input = st.text_area("Paste compositions (one per line)", height=200)
if st.button("Predict"):
    comp_lines = comp_input.strip().splitlines()
    if not comp_lines:
        st.warning("请先输入至少一行合金成分。")
    else:
        with st.spinner("模型推断中..."):
            df, errs = predict_batch(comp_lines)
        if not df.empty:
            st.success(f"成功预测 {len(df)} 条样本")
            st.dataframe(df, use_container_width=True)
            # 下载按钮
            buf = io.BytesIO()
            df.to_csv(buf, index=False)
            st.download_button("下载结果 CSV", buf.getvalue(),
                               file_name="prediction_results.csv",
                               mime="text/csv")
        if errs:
            st.error("下列成分解析失败：\n" + "\n".join(errs))


