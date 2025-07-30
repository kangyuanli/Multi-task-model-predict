# app.py  ── Streamlit 前端
import os
import math
import re
import pickle
import numpy as np
import streamlit as st
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd 

# ───────────────────────────────
# ❶ 载入模型和 Scaler（首次会略慢）
# ───────────────────────────────
@st.cache_resource(show_spinner=True)
def load_model_and_scalers(model_path: str, scaler_dir: str):
    # • 载入训练好的 MTWAE
    from mtwae_model import MTWAE           
    device = torch.device('cpu')           
    model = MTWAE(in_features=30, latent_size=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # • 载入 3 个目标变量的 scaler
    with open(os.path.join(scaler_dir, "Bs_scaler.pkl"), "rb") as f:
        Bs_scaler = pickle.load(f)
    with open(os.path.join(scaler_dir, "Hc_scaler.pkl"), "rb") as f:
        Hc_scaler = pickle.load(f)
    with open(os.path.join(scaler_dir, "Dc_scaler.pkl"), "rb") as f:
        Dc_scaler = pickle.load(f)

    return model, Bs_scaler, Hc_scaler, Dc_scaler, device

# ───────────────────────────────
# ❷ 基础工具函数
# ───────────────────────────────
PERIODIC_TABLE = [
    'Fe','B','Si','P','C','Co','Nb','Ni','Mo','Zr','Ga','Al',
    'Dy','Cu','Cr','Y','Nd','Hf','Ti','Tb','Ho','Ta','Er','Sn',
    'W','Tm','Gd','Sm','V','Pr'
]

def parse_composition_str(comp_str):
    """'Fe68.2Co17.5B13Si0.5Cu0.8' → (['Fe','Co',...], [68.2,17.5,...])"""
    pattern = r"([A-Z][a-z]?)([\d\.]+)"
    matches = re.findall(pattern, comp_str.strip())
    elements, fracs = zip(*matches) if matches else ([], [])
    return list(elements), list(map(float, fracs))

def get_element_index(element_list, periodic_table=PERIODIC_TABLE):
    return [periodic_table.index(e) for e in element_list]

def composition_to_vector(elem_list, frac_list, periodic_table=PERIODIC_TABLE):
    vec = np.zeros(len(periodic_table))
    for e, f in zip(elem_list, frac_list):
        vec[periodic_table.index(e)] = f / 100.0   # 转成原子百分比（0‑1）
    return vec

@torch.no_grad()
def predict(model, scalers, device, comp_strings):
    Bs_scl, Hc_scl, Dc_scl = scalers
    X = []
    valid_idx = []
    for i, s in enumerate(comp_strings):
        elems, fracs = parse_composition_str(s)
        if not elems:
            continue
        X.append(composition_to_vector(elems, fracs))
        valid_idx.append(i)
    if not X:
        return [None] * len(comp_strings)

    X = torch.tensor(X, dtype=torch.float32, device=device)
    _, _, pre_Bs, pre_Hc, pre_Dc = model(X)

    Bs = Bs_scl.inverse_transform(pre_Bs.cpu().numpy())
    lnHc = Hc_scl.inverse_transform(pre_Hc.cpu().numpy())
    Dc = Dc_scl.inverse_transform(pre_Dc.cpu().numpy())

    # 组装结果（保持输入顺序）
    out = [None] * len(comp_strings)
    for k, idx in enumerate(valid_idx):
        out[idx] = (float(Bs[k]), float(lnHc[k]), float(Dc[k]))
    return out

# ───────────────────────────────
# ❸ Streamlit 页面
# ───────────────────────────────
st.set_page_config(page_title="Fe‑based MG Property Predictor", layout="centered")


# 
st.title("Fe‑based Metallic Glass Property Predictor (Bs · Hc · Dc)")

st.caption(
    "Enter **one composition per line** (e.g. `Fe68.2Co17.5B13Si0.5Cu0.8`).  "
    "Supported elements: Fe, B, Si, P, C, Co, Nb, Ni, Mo, Zr, Ga, Al, … (共 30 种)."
)

# ▏折叠面板里放完整范围表
with st.expander("▶  Recommended atomic‑% ranges for each element (click to show)", expanded=False):
    st.markdown(
        "|Element|Min|Max|\n|:--|:--|:--|\n"
        "|Al|2.0|15.0|\n|B|1.0|25.0|\n|C|0.15|10.0|\n|Co|1.0|36.0|\n|Cr|1.0|6.0|\n"
        "|Cu|0.1|1.25|\n|Dy|0.5|6.72|\n|Er|1.0|7.0|\n|Fe|30.0|89.0|\n|Ga|1.0|5.0|\n"
        "|Gd|3.5|4.8|\n|Hf|1.67|12.0|\n|Ho|1.0|6.0|\n|Mo|1.0|10.0|\n|Nb|1.0|10.0|\n"
        "|Nd|3.0|3.0|\n|Ni|1.0|38.4|\n|P|1.0|14.0|\n|Pr|3.5|3.5|\n|Si|0.9|19.2|\n"
        "|Sm|3.0|3.5|\n|Sn|1.0|3.0|\n|Ta|0.75|4.0|\n|Tb|0.96|6.72|\n|Ti|1.0|5.0|\n"
        "|Tm|4.8|5.0|\n|V|2.0|2.0|\n|W|2.0|4.0|\n|Y|1.0|6.0|\n|Zr|1.0|10.0|\n"
    )

# • 侧边栏：模型与文件路径（如需切换不同模型时用）
with st.sidebar:
    st.header("Model Settings")
    model_path = st.text_input("Model weight (*.pth)", "MTWAE_latent8.pth")
    scaler_dir = st.text_input("Scaler directory", "scalers/")
    if st.button("Reload model / scalers"):
        st.cache_resource.clear()

# • 主界面：输入框
default_samples = (
    "Fe68.2Co17.5B13Si0.5Cu0.8\n"
)
user_input = st.text_area("Compositions", default_samples, height=180)
comp_list = [s for s in user_input.splitlines() if s.strip()]

# 预测按钮内部（替换原 table_data / st.dataframe 块）
if st.button("Predict"):
    with st.spinner("Loading model & predicting..."):
        mdl, *scalers_device = load_model_and_scalers(model_path, scaler_dir)
        Bs_scl, Hc_scl, Dc_scl, device = scalers_device
        results = predict(mdl, (Bs_scl, Hc_scl, Dc_scl), device, comp_list)

    # 组装成列表‑dict，便于 DataFrame 控制列顺序
    rows = []
    for idx, (comp, res) in enumerate(zip(comp_list, results), start=1):
        if res is None:
            rows.append({
                "No.": idx,
                "Composition": comp,
                "Bs (T)": "Parse Error",
                "ln(Hc) / Hc (A/m)": "–",
                "Dc (mm)": "–"
            })
        else:
            bs, lnhc, dc = res
            hc = math.exp(lnhc)
            rows.append({
                "No.": idx,
                "Composition": comp,
                "Bs (T)": f"{bs:.3f}",
                "ln(Hc) / Hc (A/m)": f"{lnhc:.3f} / {hc:.2f}",
                "Dc (mm)": f"{dc:.3f}"
            })

    df = pd.DataFrame(rows, columns=[
        "No.", "Composition", "Bs (T)", "ln(Hc) / Hc (A/m)", "Dc (mm)"
    ])
    st.dataframe(df, use_container_width=True,hide_index=True)
    st.success("Done!")


