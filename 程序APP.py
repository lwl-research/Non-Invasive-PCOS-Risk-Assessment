import os
os.environ["OMP_NUM_THREADS"] = "1"

import traceback
import numpy as np

# ========================
# 兼容性补丁
# ========================
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

import streamlit as st
import joblib
import pandas as pd
import shap
import plotly.graph_objects as go
import textwrap


# ========================
# 页面基础设置
# ========================
CONTAINER_W = 980

st.set_page_config(
    page_title="多囊卵巢综合征无创风险识别",
    layout="centered"
)

st.markdown(f"""
<style>
.main .block-container {{
  max-width: {CONTAINER_W}px;
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}}

html, body, [class*="css"] {{
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif;
}}

div[data-testid="stPlotlyChart"] {{
  padding: 12px 10px;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}}

.badge {{
  display:inline-block;
  padding:2px 10px;
  border-radius:999px;
  font-size:14px;
  font-weight:600;
  color:#fff;
  margin-left:10px;
  vertical-align:middle;
}}

.stButton > button {{
  width: 100%;
  border-radius: 10px;
  font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)


# ========================
# 路径设置
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = "Stacking_RF_LightGBM.pkl"

# 局部解释图只展示前 N 个贡献最大的变量，其余变量合并为一项
TOP_N_CONTRIBUTIONS = 6

BACKGROUND_CANDIDATES = [
    "shap_background.csv",
    "shap_background.xlsx",
    "shap_background.xls",
    "shap_background(1).csv"
]


# ========================
# 分类变量映射
# 注意：编码值必须和训练数据一致
# ========================
OPTION_MAP = {
    "Age at menarche": {
        "<11岁": 1,
        "11–15岁": 2,
        ">15岁": 3
    },

    "Menstrual cycle regularity": {
        "频发（<21天）": 1,
        "周期不规律": 2,
        "规律（21–35天）": 3,
        "稀发或闭经（>35天或6个月以上不来）": 4
    },

    "Hair loss": {
        "否": 0,
        "是": 1
    }
}

VALUE_TO_TEXT = {
    feat: {v: k for k, v in mapping.items()}
    for feat, mapping in OPTION_MAP.items()
}


# ========================
# 变量显示名和默认值
# 外层英文变量名不要改，因为模型内部使用这些列名
# ========================
FEATURE_META = {
    "Age": {
        "label": "年龄（岁）",
        "type": "numerical",
        "default": 29.50,
        "step": 0.01
    },

    "Neck circumference": {
        "label": "颈围（cm）",
        "type": "numerical",
        "default": 31.38,
        "step": 0.01
    },

    "Waist circumference": {
        "label": "腰围（cm）",
        "type": "numerical",
        "default": 74.72,
        "step": 0.01
    },

    "Hip circumference": {
        "label": "臀围（cm）",
        "type": "numerical",
        "default": 94.29,
        "step": 0.01
    },

    "Systolic blood pressure": {
        "label": "收缩压（mmHg）",
        "type": "numerical",
        "default": 111.14,
        "step": 0.01
    },

    "Diastolic blood pressure": {
        "label": "舒张压（mmHg）",
        "type": "numerical",
        "default": 71.98,
        "step": 0.01
    },

    "Skeletal muscle mass": {
        "label": "骨骼肌量（kg）",
        "type": "numerical",
        "default": 21.43,
        "step": 0.01
    },

    "Percent body fat": {
        "label": "体脂率（%）",
        "type": "numerical",
        "default": 35.09,
        "step": 0.01
    },

    "Body mass index": {
        "label": "体重指数（kg/m²）",
        "type": "numerical",
        "default": 23.54,
        "step": 0.01
    },

    "Age at menarche": {
        "label": "初潮年龄",
        "type": "categorical",
        "default": "11–15岁"
    },

    "Menstrual cycle regularity": {
        "label": "月经周期规律性",
        "type": "categorical",
        "default": "规律（21–35天）"
    },

    "Hirsutism score": {
        "label": "多毛评分",
        "type": "numerical",
        "default": 3.15,
        "step": 1.00
    },

    "Hair loss": {
        "label": "是否脱发",
        "type": "categorical",
        "default": "否"
    },

    "Acne score": {
        "label": "痤疮评分",
        "type": "numerical",
        "default": 3.32,
        "step": 1.00
    }
}


# ========================
# 痤疮综合分级评分
# 综合分值 = Σ（区域因素分值 × 皮损分值）
# 总分范围：0–44
# ========================
ACNE_REGIONS = [
    "前额",
    "右颊部",
    "左颊",
    "鼻部",
    "下颌区",
    "胸及上背部"
]

ACNE_OPTIONS = {
    "无皮损": 0,
    "皮损>1个粉刺": 1,
    "皮损>1个丘疹": 2,
    "皮损>1个脓疱": 3,
    "皮损>1个结节或囊肿": 4
}

ACNE_WEIGHTS = {
    "前额": 2,
    "右颊部": 2,
    "左颊": 2,
    "鼻部": 1,
    "下颌区": 1,
    "胸及上背部": 3
}


def calculate_acne_score(acne_selections):
    total = 0
    for region, choice in acne_selections.items():
        lesion_score = ACNE_OPTIONS[choice]
        factor_score = ACNE_WEIGHTS[region]
        total += factor_score * lesion_score
    return float(total)


def classify_acne_score(score):
    if score == 0:
        return "无痤疮"
    elif 1 <= score <= 18:
        return "轻度"
    elif 19 <= score <= 30:
        return "中度"
    elif 31 <= score <= 38:
        return "重度"
    else:
        return "特重"


# ========================
# 改良 Ferriman-Gallwey 9部位评分
# 总分范围：0–36
# ========================
HIRSUTISM_ITEMS = {
    "上腹部": [
        "0分：无毛",
        "1分：中线少许毛",
        "2分：较多毛但仍在中线",
        "3分：毛覆盖1/2",
        "4分：毛覆盖全部"
    ],
    "下腹部": [
        "0分：无毛",
        "1分：中线少许毛",
        "2分：中线毛呈条状",
        "3分：中线毛呈带状",
        "4分：中线毛呈倒V形状"
    ],
    "下背部": [
        "0分：无毛",
        "1分：骶部一簇毛",
        "2分：略向两侧伸展",
        "3分：覆盖表面3/4",
        "4分：完全覆盖"
    ],
    "上唇": [
        "0分：无毛",
        "1分：外侧毛少许",
        "2分：外侧小胡须",
        "3分：胡须向内延伸未达中线",
        "4分：胡须延伸至中线"
    ],
    "大腿": [
        "0分：无毛",
        "1分：毛稀疏，未超过表面1/4",
        "2分：超过1/4，但未完全覆盖",
        "3分：完全覆盖，毛淡",
        "4分：完全覆盖，毛浓"
    ],
    "上背": [
        "0分：无毛",
        "1分：少许稀疏毛",
        "2分：较多但仍分散",
        "3分：完全覆盖，淡",
        "4分：完全覆盖，浓"
    ],
    "胸部": [
        "0分：无毛",
        "1分：乳晕周围毛",
        "2分：乳晕周围毛伴中线毛",
        "3分：毛发融合覆盖3/4面积",
        "4分：完全覆盖"
    ],
    "上臂": [
        "0分：无毛",
        "1分：毛稀疏，未超过表面1/4",
        "2分：超过1/4，但未完全覆盖",
        "3分：完全覆盖，毛淡",
        "4分：完全覆盖，毛浓"
    ],
    "下颌": [
        "0分：无毛",
        "1分：少许散在毛",
        "2分：分散的毛有小聚集",
        "3分：完全覆盖，淡毛",
        "4分：完全覆盖，浓毛"
    ]
}


def calculate_hirsutism_score(hirsutism_scores):
    return float(sum(hirsutism_scores.values()))


def classify_hirsutism_score(hirsutism_scores):
    total = calculate_hirsutism_score(hirsutism_scores)

    three_site_total = (
        hirsutism_scores.get("上唇", 0)
        + hirsutism_scores.get("下腹部", 0)
        + hirsutism_scores.get("大腿", 0)
    )

    if total >= 4 or three_site_total >= 2:
        return "达到多毛判定参考"
    return "未达到多毛判定参考"


# ========================
# 模型加载工具函数
# ========================
def load_joblib_with_clear_error(path, model_label):
    try:
        return joblib.load(path)

    except ModuleNotFoundError as e:
        st.error(f"加载模型失败：{model_label}")
        st.code(
            f"Model file: {os.path.basename(str(path))}\n"
            f"Missing module: {e.name}\n"
            f"Error: {repr(e)}"
        )
        st.stop()

    except Exception:
        st.error(f"加载模型失败：{model_label}")
        st.code(traceback.format_exc())
        st.stop()


def extract_predictor(obj):
    if hasattr(obj, "predict_proba"):
        return obj

    if isinstance(obj, dict):
        preferred_keys = [
            "final_model",
            "final_pipeline",
            "model",
            "best_model",
            "estimator",
            "classifier",
            "clf"
        ]

        for key in preferred_keys:
            if key in obj and hasattr(obj[key], "predict_proba"):
                return obj[key]

        for value in obj.values():
            if hasattr(value, "predict_proba"):
                return value

    raise ValueError("未能从 pkl 文件中提取支持 predict_proba 的模型。")


def resolve_model_path(path_like):
    path_like = str(path_like)

    if os.path.isabs(path_like) and os.path.exists(path_like):
        return path_like

    candidate_1 = os.path.join(BASE_DIR, path_like)
    if os.path.exists(candidate_1):
        return candidate_1

    candidate_2 = os.path.join(BASE_DIR, os.path.basename(path_like))
    if os.path.exists(candidate_2):
        return candidate_2

    raise FileNotFoundError(f"未找到基模型文件: {path_like}")


@st.cache_resource
def load_model_bundle(model_file):
    model_path = os.path.join(BASE_DIR, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到集成模型文件: {model_path}")

    obj = load_joblib_with_clear_error(model_path, model_file)

    if hasattr(obj, "predict_proba"):
        feature_cols = getattr(obj, "feature_names_in_", None)

        if feature_cols is None:
            raise ValueError("单模型中没有 feature_names_in_，请手动提供 MODEL_FEATURES。")

        return {
            "mode": "single",
            "model_name": os.path.splitext(model_file)[0],
            "model": obj,
            "feature_cols": list(feature_cols),
            "threshold": 0.5
        }

    if not isinstance(obj, dict):
        raise ValueError("模型既不是单模型，也不是集成模型字典。")

    members = obj.get("members", None)
    feature_cols = obj.get("feature_cols", None)
    ensemble_type = obj.get("ensemble_type", "")
    model_name = obj.get("model_name", os.path.splitext(model_file)[0])
    threshold = float(obj.get("threshold", 0.5))

    if members is None:
        raise ValueError("集成模型字典中缺少 members。")

    if feature_cols is None:
        raise ValueError("集成模型字典中缺少 feature_cols。")

    base_model_files = obj.get("base_model_files", {})
    base_models = {}

    for member in members:
        if member not in base_model_files:
            raise ValueError(f"base_model_files 中缺少成员模型 {member} 的文件路径。")

        base_path = resolve_model_path(base_model_files[member])
        base_obj = load_joblib_with_clear_error(base_path, f"base model: {member}")
        base_models[member] = extract_predictor(base_obj)

    if "Stacking" in str(ensemble_type) or str(model_name).startswith("Stacking"):
        meta_model = obj.get("meta_model", None)

        if meta_model is None:
            raise ValueError("Stacking 模型字典中缺少 meta_model。")

        return {
            "mode": "stacking",
            "model_name": model_name,
            "members": members,
            "base_models": base_models,
            "meta_model": meta_model,
            "feature_cols": list(feature_cols),
            "threshold": threshold
        }

    if "Weighted" in str(ensemble_type) or str(model_name).startswith("WeightedVoting"):
        weights = obj.get("weights", None)

        if weights is None:
            raise ValueError("WeightedVoting 模型字典中缺少 weights。")

        return {
            "mode": "weighted",
            "model_name": model_name,
            "members": members,
            "base_models": base_models,
            "weights": weights,
            "feature_cols": list(feature_cols),
            "threshold": threshold
        }

    raise ValueError(f"暂不支持的集成模型类型: {ensemble_type}")


# ========================
# 背景数据读取
# ========================
def find_background_path():
    for fname in BACKGROUND_CANDIDATES:
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            return path
    return None


def read_table(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"不支持的背景数据格式: {ext}")


# ========================
# 加载模型
# ========================
bundle = load_model_bundle(MODEL_FILE)

MODEL_FEATURES = bundle["feature_cols"]
FIXED_THRESHOLD = bundle["threshold"]

CATEGORICAL_COLS = [c for c in MODEL_FEATURES if c in OPTION_MAP]
NUMERICAL_COLS = [c for c in MODEL_FEATURES if c not in CATEGORICAL_COLS]


# ========================
# 输入整理
# ========================
def prepare_input_df(data_like):
    if isinstance(data_like, pd.DataFrame):
        df = data_like.copy()
    else:
        arr = np.asarray(data_like)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        df = pd.DataFrame(arr, columns=MODEL_FEATURES)

    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"输入数据缺少变量: {missing}")

    df = df[MODEL_FEATURES].copy()

    for col in CATEGORICAL_COLS:
        df[col] = np.rint(pd.to_numeric(df[col], errors="raise")).astype(int)

    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def predict_positive_proba(data_like):
    df = prepare_input_df(data_like)

    if bundle["mode"] == "single":
        return bundle["model"].predict_proba(df)[:, 1]

    base_probs = []

    for member in bundle["members"]:
        p = bundle["base_models"][member].predict_proba(df)[:, 1]
        base_probs.append(p)

    base_probs = np.column_stack(base_probs)

    if bundle["mode"] == "stacking":
        return bundle["meta_model"].predict_proba(base_probs)[:, 1]

    if bundle["mode"] == "weighted":
        weights = np.array([float(bundle["weights"][m]) for m in bundle["members"]])
        weights = weights / weights.sum()
        return np.dot(base_probs, weights)

    raise ValueError("未知模型模式。")


# ========================
# 加载 SHAP 背景数据
# ========================
@st.cache_data
def load_background_data():
    bg_path = find_background_path()

    if bg_path is None:
        return None

    bg = read_table(bg_path)

    missing = [c for c in MODEL_FEATURES if c not in bg.columns]
    if missing:
        raise ValueError(f"SHAP 背景数据缺少以下列: {missing}")

    bg = prepare_input_df(bg[MODEL_FEATURES])

    if len(bg) > 100:
        bg = bg.sample(n=100, random_state=42)

    return bg


background_df = load_background_data()


# ========================
# 标签与格式
# ========================
def classify_prediction(p: float, threshold: float):
    if p >= threshold:
        return "预测可能性较高", "#C62828"
    return "预测可能性较低", "#2E7D32"


def get_display_name(feature):
    if feature == "Other features":
        return "其他特征"
    return FEATURE_META.get(feature, {}).get("label", feature)


def format_feature_value(feature, value):
    if feature in VALUE_TO_TEXT:
        try:
            value_int = int(round(float(value)))
        except Exception:
            return str(value)

        return VALUE_TO_TEXT[feature].get(value_int, str(value_int))

    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def format_contribution(x):
    if abs(x) < 0.005:
        return "0.00"
    return f"{x:+.2f}"


# ========================
# SHAP explainer
# ========================
def get_shap_explainer():
    if background_df is None:
        return None

    if "kernel_shap_explainer" not in st.session_state:
        st.session_state["kernel_shap_explainer"] = shap.KernelExplainer(
            predict_positive_proba,
            background_df,
            link="identity"
        )

    return st.session_state["kernel_shap_explainer"]


def compute_kernel_shap_probability(X_one_row):
    explainer = get_shap_explainer()

    if explainer is None:
        return None, None

    shap_values = explainer.shap_values(X_one_row, nsamples=200)

    if isinstance(shap_values, list):
        shap_values = np.asarray(shap_values[0])
    else:
        shap_values = np.asarray(shap_values)

    if shap_values.ndim == 2:
        local_vals = shap_values[0]
    else:
        local_vals = shap_values.reshape(-1)

    base_value = explainer.expected_value

    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).reshape(-1)[0]

    return local_vals, float(base_value)


# ========================
# 局部贡献图
# ========================
def summarize_contributions_for_plot(df_sorted, top_n=TOP_N_CONTRIBUTIONS):
    df_sorted = df_sorted.copy().reset_index(drop=True)

    if len(df_sorted) <= top_n:
        return df_sorted

    top_df = df_sorted.iloc[:top_n].copy()
    other_df = df_sorted.iloc[top_n:].copy()

    other_contribution = float(other_df["dpp"].sum())

    other_row = pd.DataFrame({
        "feature": ["Other features"],
        "value": [np.nan],
        "value_text": ["合并"],
        "dpp": [other_contribution]
    })

    return pd.concat([top_df, other_row], ignore_index=True)


def plot_pp_bar(df_plot):
    df_plot = df_plot.copy()

    labels = []
    for f, vtxt in zip(df_plot["feature"], df_plot["value_text"]):
        if f == "Other features":
            label = "其他特征（合并）"
        else:
            label = f"{get_display_name(f)} = {vtxt}"
        labels.append(textwrap.fill(label, width=30))

    x_vals = df_plot["dpp"].to_numpy()
    colors = np.where(x_vals >= 0, "#E45756", "#4C78A8")

    texts, textpos, textcolor = [], [], []

    for x in x_vals:
        texts.append(format_contribution(x))

        if x < 0:
            textpos.append("inside")
            textcolor.append("white")
        else:
            if abs(x) >= 1:
                textpos.append("inside")
                textcolor.append("white")
            else:
                textpos.append("outside")
                textcolor.append("black")

    fig = go.Figure(go.Bar(
        y=labels[::-1],
        x=x_vals[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=texts[::-1],
        texttemplate="%{text}",
        textposition=textpos[::-1],
        insidetextanchor="end",
        textfont=dict(color=textcolor[::-1], size=14),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>贡献值：%{x:+.2f} 个百分点<extra></extra>",
    ))

    fig.update_layout(
        height=max(260, 38 * len(labels) + 80),
        margin=dict(l=300, r=80, t=12, b=12),
        font=dict(family="Microsoft YaHei, PingFang SC, Arial", size=16),
        yaxis=dict(
            title="",
            type="category",
            tickfont=dict(size=14),
            automargin=True
        ),
        xaxis=dict(
            title="对模型预测概率的近似贡献（百分点）",
            zeroline=True,
            zerolinewidth=1.2,
            zerolinecolor="#B0BEC5",
            showgrid=True,
            gridcolor="#EFEFEF",
            automargin=True
        ),
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        uniformtext_minsize=12,
        uniformtext_mode="hide"
    )

    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color="#B0BEC5",
        line_width=1
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


# ========================
# 页面标题
# ========================
st.title("多囊卵巢综合征无创风险识别")
st.caption("基于无创临床指标估计个体化 PCOS 风险。")


# ========================
# 输入表单
# ========================
with st.form("prediction_form", clear_on_submit=False):

    st.markdown("### 一、基本信息与月经情况")

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        age = st.number_input(
            FEATURE_META["Age"]["label"],
            min_value=0.0,
            value=float(FEATURE_META["Age"]["default"]),
            step=float(FEATURE_META["Age"]["step"]),
            format="%.2f",
            key="Age_input"
        )

        age_menarche = st.selectbox(
            "您的月经初潮年龄？",
            options=list(OPTION_MAP["Age at menarche"].keys()),
            index=list(OPTION_MAP["Age at menarche"].keys()).index(
                FEATURE_META["Age at menarche"]["default"]
            ),
            key="Age at menarche_input"
        )

        hair_loss = st.selectbox(
            "是否存在明显脱发或头顶部毛发稀疏？",
            options=list(OPTION_MAP["Hair loss"].keys()),
            index=list(OPTION_MAP["Hair loss"].keys()).index(
                FEATURE_META["Hair loss"]["default"]
            ),
            key="Hair loss_input"
        )

    with col_right:
        menstrual_cycle = st.selectbox(
            "您的月经周期情况？",
            options=list(OPTION_MAP["Menstrual cycle regularity"].keys()),
            index=list(OPTION_MAP["Menstrual cycle regularity"].keys()).index(
                FEATURE_META["Menstrual cycle regularity"]["default"]
            ),
            key="Menstrual cycle regularity_input"
        )
        st.caption("请根据近期通常情况选择。闭经在当前模型输入中按稀发或>35天类别处理。")

    st.markdown("### 二、体格测量与身体成分")

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        height_cm = st.number_input(
            "身高（cm，用于自动计算体重指数）",
            min_value=1.0,
            value=160.00,
            step=0.01,
            format="%.2f",
            key="height_cm_input"
        )

        weight_kg = st.number_input(
            "体重（kg，用于自动计算体重指数）",
            min_value=1.0,
            value=60.26,
            step=0.01,
            format="%.2f",
            key="weight_kg_input"
        )

        bmi_value = weight_kg / ((height_cm / 100.0) ** 2)

        st.info(f"系统计算的体重指数（BMI）：{bmi_value:.2f} kg/m²")

        waist = st.number_input(
            FEATURE_META["Waist circumference"]["label"],
            min_value=0.0,
            value=float(FEATURE_META["Waist circumference"]["default"]),
            step=float(FEATURE_META["Waist circumference"]["step"]),
            format="%.2f",
            key="Waist circumference_input"
        )

        sbp = st.number_input(
            FEATURE_META["Systolic blood pressure"]["label"],
            min_value=0.0,
            value=float(FEATURE_META["Systolic blood pressure"]["default"]),
            step=float(FEATURE_META["Systolic blood pressure"]["step"]),
            format="%.2f",
            key="Systolic blood pressure_input"
        )

        skeletal_muscle = st.number_input(
            FEATURE_META["Skeletal muscle mass"]["label"],
            min_value=0.0,
            value=float(FEATURE_META["Skeletal muscle mass"]["default"]),
            step=float(FEATURE_META["Skeletal muscle mass"]["step"]),
            format="%.2f",
            key="Skeletal muscle mass_input"
        )

    with col_right:
        neck = st.number_input(
            FEATURE_META["Neck circumference"]["label"],
            min_value=0.0,
            value=float(FEATURE_META["Neck circumference"]["default"]),
            step=float(FEATURE_META["Neck circumference"]["step"]),
            format="%.2f",
            key="Neck circumference_input"
        )

        hip = st.number_input(
            FEATURE_META["Hip circumference"]["label"],
            min_value=0.0,
            value=float(FEATURE_META["Hip circumference"]["default"]),
            step=float(FEATURE_META["Hip circumference"]["step"]),
            format="%.2f",
            key="Hip circumference_input"
        )

        dbp = st.number_input(
            FEATURE_META["Diastolic blood pressure"]["label"],
            min_value=0.0,
            value=float(FEATURE_META["Diastolic blood pressure"]["default"]),
            step=float(FEATURE_META["Diastolic blood pressure"]["step"]),
            format="%.2f",
            key="Diastolic blood pressure_input"
        )

        body_fat = st.number_input(
            FEATURE_META["Percent body fat"]["label"],
            min_value=0.0,
            value=float(FEATURE_META["Percent body fat"]["default"]),
            step=float(FEATURE_META["Percent body fat"]["step"]),
            format="%.2f",
            key="Percent body fat_input"
        )

    st.markdown("### 三、症状量表评分")

    with st.expander("痤疮综合分级评分（点击展开填写）", expanded=False):
        st.caption(
            "请根据各区域目前最严重的皮损情况进行选择。"
            "系统将按“区域因素分值 × 皮损分值”自动计算痤疮综合分值。"
        )

        acne_selections = {}

        for region in ACNE_REGIONS:
            acne_selections[region] = st.selectbox(
                f"{region}目前最严重的皮损情况",
                options=list(ACNE_OPTIONS.keys()),
                index=0,
                key=f"acne_{region}"
            )

        acne_total = calculate_acne_score(acne_selections)

        st.info(
            f"系统计算的痤疮评分：{acne_total:.0f} / 44，"
            f"分级：{classify_acne_score(acne_total)}"
        )

    with st.expander("多毛评分（改良 Ferriman-Gallwey 9部位评分，点击展开填写）", expanded=False):
        st.caption(
            "请根据9个部位的毛发生长情况选择0–4分。"
            "系统将自动计算多毛总评分。"
        )

        hirsutism_scores = {}

        for item, descriptions in HIRSUTISM_ITEMS.items():
            score = st.select_slider(
                f"{item}毛发生长情况",
                options=[0, 1, 2, 3, 4],
                value=0,
                format_func=lambda x, desc=descriptions: desc[x],
                key=f"hirsutism_{item}"
            )
            hirsutism_scores[item] = int(score)

        hirsutism_total = calculate_hirsutism_score(hirsutism_scores)
        hirsutism_level = classify_hirsutism_score(hirsutism_scores)

        st.info(
            f"系统计算的多毛评分：{hirsutism_total:.0f} / 36，"
            f"{hirsutism_level}"
        )

    submitted = st.form_submit_button("开始预测", type="primary")


# ========================
# 预测与解释
# ========================
if submitted:
    form_values = {}

    for feature in MODEL_FEATURES:
        if feature == "Age":
            form_values[feature] = float(age)

        elif feature == "Age at menarche":
            form_values[feature] = int(OPTION_MAP[feature][age_menarche])

        elif feature == "Menstrual cycle regularity":
            form_values[feature] = int(OPTION_MAP[feature][menstrual_cycle])

        elif feature == "Hair loss":
            form_values[feature] = int(OPTION_MAP[feature][hair_loss])

        elif feature == "Body mass index":
            form_values[feature] = float(bmi_value)

        elif feature == "Neck circumference":
            form_values[feature] = float(neck)

        elif feature == "Waist circumference":
            form_values[feature] = float(waist)

        elif feature == "Hip circumference":
            form_values[feature] = float(hip)

        elif feature == "Systolic blood pressure":
            form_values[feature] = float(sbp)

        elif feature == "Diastolic blood pressure":
            form_values[feature] = float(dbp)

        elif feature == "Skeletal muscle mass":
            form_values[feature] = float(skeletal_muscle)

        elif feature == "Percent body fat":
            form_values[feature] = float(body_fat)

        elif feature == "Acne score":
            form_values[feature] = float(acne_total)

        elif feature == "Hirsutism score":
            form_values[feature] = float(hirsutism_total)

        else:
            raise ValueError(f"当前界面未配置模型变量: {feature}")

    X = pd.DataFrame(
        [[form_values[col] for col in MODEL_FEATURES]],
        columns=MODEL_FEATURES
    )

    X = prepare_input_df(X)

    p1 = float(predict_positive_proba(X)[0])
    pred_label, pred_color = classify_prediction(p1, FIXED_THRESHOLD)

    st.markdown(
        f"""
        <div style='font-family:Microsoft YaHei, PingFang SC, Arial; font-size:20px;'>
          <b>模型预测的 PCOS 风险概率：{p1 * 100:.2f}%</b>
          <span class="badge" style="background:{pred_color};">{pred_label}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("本结果用于 PCOS 风险识别和初步筛查参考，不作为临床诊断结论。")

    if background_df is None:
        st.info(
            "预测结果已生成。若需显示局部特征贡献，请将 shap_background.csv、"
            "shap_background.xlsx 或 shap_background.xls 放入本应用同一文件夹。"
        )

    else:
        with st.spinner("正在计算局部特征贡献..."):
            shap_vals, _ = compute_kernel_shap_probability(X)

        if shap_vals is None:
            st.info("预测结果已生成，但当前无法计算局部特征贡献。")

        else:
            feat_vals = X.iloc[0].to_numpy()
            dpp = shap_vals * 100.0

            order = np.argsort(-np.abs(dpp), kind="mergesort")

            ordered_features = X.columns.to_numpy()[order]
            ordered_values = feat_vals[order]

            df_sorted = pd.DataFrame({
                "feature": ordered_features,
                "value": ordered_values,
                "value_text": [
                    format_feature_value(f, v)
                    for f, v in zip(ordered_features, ordered_values)
                ],
                "dpp": dpp[order],
            })

            df_for_plot = summarize_contributions_for_plot(
                df_sorted,
                top_n=TOP_N_CONTRIBUTIONS
            )
            plot_pp_bar(df_for_plot)

            with st.expander("查看全部特征贡献"):
                table = df_sorted[["feature", "value_text", "dpp"]].copy()
                table["feature"] = table["feature"].map(get_display_name)
                table.columns = [
                    "特征",
                    "取值",
                    "贡献值（百分点）"
                ]
                table.insert(0, "排序", np.arange(1, len(table) + 1))

                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True
                )