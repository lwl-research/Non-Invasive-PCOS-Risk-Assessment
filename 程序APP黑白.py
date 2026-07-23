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
CONTAINER_W = 1180

st.set_page_config(
    page_title="Non-Invasive PCOS Risk Assessment",
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
  color: #111111;
  background: #FFFFFF;
}}

/* 页面整体黑白化 */
section.main {{
  background: #FFFFFF;
}}

h1, h2, h3, h4, p, span, label, div {{
  color: #111111;
}}
h1 {{
  font-size: 2.2rem !important;
  font-weight: 700 !important;
}}

h2, h3 {{
  font-size: 1.45rem !important;
  font-weight: 700 !important;
}}

p, span, label, div {{
  font-size: 17.5px;
}}

div[data-testid="stCaptionContainer"],
div[data-testid="stMarkdownContainer"] p {{
  font-size: 17px !important;
}}

input {{
  font-size: 17.5px !important;
  padding-top: 0.45rem !important;
  padding-bottom: 0.45rem !important;
}}

div[data-baseweb="select"] span {{
  font-size: 17.5px !important;
}}


/* 输入框、选择框、表格等统一黑白边框 */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
textarea,
input {{
  border-color: #222222 !important;
  color: #111111 !important;
  background-color: #FFFFFF !important;
}}

div[data-testid="stPlotlyChart"] {{
  padding: 12px 10px;
  background: #ffffff;
  border: 1.4px solid #222222;
  border-radius: 8px;
  box-shadow: none;
}}

.badge {{
  display:inline-block;
  padding:2px 10px;
  border-radius:999px;
  font-size:15px;
  font-weight:600;
  color:#fff;
  margin-left:10px;
  vertical-align:middle;
  border: 1px solid #111111;
}}

.stButton > button {{
  width: 100%;
  border-radius: 8px;
  font-weight: 600;
  background-color: #FFFFFF !important;
  color: #111111 !important;
  border: 1.5px solid #111111 !important;
}}

.stButton > button:hover {{
  background-color: #F2F2F2 !important;
  color: #111111 !important;
  border: 1.5px solid #111111 !important;
}}

/* 信息提示框黑白化 */
div[data-testid="stInfo"],
div[data-testid="stAlert"] {{
  background-color: #F5F5F5 !important;
  color: #111111 !important;
  border: 1px solid #222222 !important;
}}

/* 展开框黑白化 */
details {{
  border: 1px solid #222222 !important;
  border-radius: 8px !important;
  background-color: #FFFFFF !important;
}}

/* 隐藏标题旁边的锚点链接图标 */
[data-testid="stMarkdownContainer"] h1 a,
[data-testid="stMarkdownContainer"] h2 a,
[data-testid="stMarkdownContainer"] h3 a {{
    display: none !important;
}}
</style>
""", unsafe_allow_html=True)


# ========================
# 路径设置
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = "Stacking_RF_LightGBM.pkl"

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
        "＜11岁": 1,
        "11–15岁": 2,
        "＞15岁": 3
    },

    "Menstrual cycle regularity": {
        "月经频发（周期＜21天）": 1,
        "月经规律（周期21–35天）": 3,
        "月经稀发（周期＞35天）": 4,
        "月经周期不规则（周期长短不定，可＜21天或＞35天）": 2
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
# 页面显示名与默认值
# 外层英文变量名不要改，因为模型内部使用这些列名
# ========================
FEATURE_META = {
    "Age": {
        "label": "年龄（岁）",
        "type": "numerical",
        "default": 25,
        "step": 1
    },

    "Neck circumference": {
        "label": "颈围（cm）",
        "type": "numerical",
        "default": 31.0,
        "step": 0.1
    },

    "Waist circumference": {
        "label": "腰围（cm）",
        "type": "numerical",
        "default": 70.0,
        "step": 0.1
    },

    "Hip circumference": {
        "label": "臀围（cm）",
        "type": "numerical",
        "default": 90.0,
        "step": 0.1
    },

    "Systolic blood pressure": {
        "label": "收缩压（mmHg）",
        "type": "numerical",
        "default": 110,
        "step": 1
    },

    "Diastolic blood pressure": {
        "label": "舒张压（mmHg）",
        "type": "numerical",
        "default": 70,
        "step": 1
    },

    "Skeletal muscle mass": {
        "label": "骨骼肌量（kg）",
        "type": "numerical",
        "default": 22.0,
        "step": 0.1
    },

    "Percent body fat": {
        "label": "体脂率（%）",
        "type": "numerical",
        "default": 28.0,
        "step": 0.1
    },

    "Body mass index": {
        "label": "BMI",
        "type": "numerical",
        "default": 21.48,
        "step": 0.01
    },

    "Age at menarche": {
        "label": "月经初潮年龄",
        "type": "categorical",
        "default": "11–15岁"
    },

    "Menstrual cycle regularity": {
        "label": "月经周期情况",
        "type": "categorical",
        "default": "月经规律（周期21–35天）"
    },

    "Hirsutism score": {
        "label": "多毛评分",
        "type": "numerical",
        "default": 0,
        "step": 1
    },

    "Hair loss": {
        "label": "是否存在头顶部毛发稀疏或明显脱发？",
        "type": "categorical",
        "default": "否"
    },

    "Acne score": {
        "label": "痤疮评分",
        "type": "numerical",
        "default": 0,
        "step": 1
    }
}


# ========================
# 初始化输入状态
# 作用：避免点击“开始评估”后输入值返回默认值
# ========================
DEFAULT_INPUTS = {
    # 专利截图示例：设置为“稍高风险倾向”的模拟个体参数
    "Age_input": 27,
    "Age at menarche_input": "＞15岁",
    "Menstrual cycle regularity_input": "月经稀发（周期＞35天）",
    "Hair loss_input": "是",

    "height_cm_input": 160.0,
    "weight_kg_input": 70.0,

    "Neck circumference_input": 35.0,
    "Waist circumference_input": 86.0,
    "Hip circumference_input": 98.0,
    "Systolic blood pressure_input": 125,
    "Diastolic blood pressure_input": 82,
    "Skeletal muscle mass_input": 22.0,
    "Percent body fat_input": 36.0,
}

NUMERIC_INPUT_LIMITS = {
    "Age_input": (10, 60),

    "height_cm_input": (120.0, 220.0),
    "weight_kg_input": (25.0, 200.0),

    "Neck circumference_input": (20.0, 60.0),
    "Waist circumference_input": (40.0, 180.0),
    "Hip circumference_input": (50.0, 200.0),

    "Systolic blood pressure_input": (70, 250),
    "Diastolic blood pressure_input": (40, 150),

    "Skeletal muscle mass_input": (5.0, 80.0),
    "Percent body fat_input": (3.0, 80.0),
}

INTEGER_INPUT_KEYS = {
    "Age_input",
    "Systolic blood pressure_input",
    "Diastolic blood pressure_input"
}

SELECT_INPUT_OPTIONS = {
    "Age at menarche_input": list(OPTION_MAP["Age at menarche"].keys()),
    "Menstrual cycle regularity_input": list(OPTION_MAP["Menstrual cycle regularity"].keys()),
    "Hair loss_input": list(OPTION_MAP["Hair loss"].keys()),
}


def init_session_state():
    for k, v in DEFAULT_INPUTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    for k, options in SELECT_INPUT_OPTIONS.items():
        if st.session_state.get(k) not in options:
            st.session_state[k] = DEFAULT_INPUTS[k]

    for k, (lo, hi) in NUMERIC_INPUT_LIMITS.items():
        try:
            v = float(st.session_state.get(k, DEFAULT_INPUTS[k]))
        except Exception:
            v = float(DEFAULT_INPUTS[k])

        if v < lo:
            v = lo
        if v > hi:
            v = hi

        if k in INTEGER_INPUT_KEYS:
            st.session_state[k] = int(round(v))
        else:
            st.session_state[k] = float(v)

    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False

def text_number_input(label, state_key, min_value, max_value, is_int=False, digits=1):
    """
    专利截图版数值输入框：
    使用普通文本框替代 st.number_input，避免截图中出现右侧 “- / +” 控件。
    仅改变界面呈现方式，不改变后续模型输入逻辑。
    """
    default_value = st.session_state.get(state_key, DEFAULT_INPUTS[state_key])

    if is_int:
        default_text = str(int(round(float(default_value))))
    else:
        default_text = f"{float(default_value):.{digits}f}"

    raw_value = st.text_input(
        label,
        value=default_text,
        key=f"{state_key}_text"
    )

    try:
        value = float(str(raw_value).strip().replace(",", ""))
    except Exception:
        st.warning(f"{label}请输入有效数字，当前使用默认值。")
        value = float(default_value)

    if value < min_value:
        st.warning(f"{label}不能低于 {min_value}，已按下限处理。")
        value = float(min_value)

    if value > max_value:
        st.warning(f"{label}不能高于 {max_value}，已按上限处理。")
        value = float(max_value)

    if is_int:
        value = int(round(value))
    else:
        value = float(value)

    st.session_state[state_key] = value
    return value



# ========================
# 痤疮综合分级评分
# 综合分值 = Σ（区域因素分值 × 皮损分值）
# 总分范围：0–44
# ========================
ACNE_REGIONS = [
    "前额",
    "右颊",
    "左颊",
    "鼻部",
    "下颌",
    "胸及上背部"
]

ACNE_OPTIONS = {
    "无皮损": 0,
    "皮损＞1个粉刺": 1,
    "皮损＞1个丘疹": 2,
    "皮损＞1个脓疱": 3,
    "皮损＞1个结节或囊肿": 4
}

ACNE_WEIGHTS = {
    "前额": 2,
    "右颊": 2,
    "左颊": 2,
    "鼻部": 1,
    "下颌": 1,
    "胸及上背部": 3
}

# 专利截图示例：轻中度痤疮倾向，避免全部为 0 导致页面过于“正常”
ACNE_DEFAULT_SELECTIONS = {
    "前额": "皮损＞1个丘疹",
    "右颊": "皮损＞1个丘疹",
    "左颊": "皮损＞1个粉刺",
    "鼻部": "皮损＞1个粉刺",
    "下颌": "皮损＞1个丘疹",
    "胸及上背部": "无皮损"
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
# 专利截图示例：轻度多毛倾向，便于展示较高风险识别结果
HIRSUTISM_DEFAULT_SCORES = {
    "上唇": 1,
    "下颌": 1,
    "胸部": 0,
    "上腹部": 1,
    "下腹部": 2,
    "上背": 0,
    "下背部": 0,
    "上臂": 0,
    "大腿": 1
}


HIRSUTISM_ITEMS = {
    "上唇": [
        "无毛",
        "外侧毛少许",
        "外侧小胡须",
        "胡须向内延伸未达中线",
        "胡须延伸至中线"
    ],
    "下颌": [
        "无毛",
        "少许散在毛",
        "分散的毛有小聚集",
        "完全覆盖，淡毛",
        "完全覆盖，浓毛"
    ],
    "胸部": [
        "无毛",
        "乳晕周围毛",
        "乳晕周围毛伴中线毛",
        "毛发融合覆盖3/4面积",
        "完全覆盖"
    ],
    "上腹部": [
        "无毛",
        "中线少许毛",
        "较多毛但仍在中线",
        "毛覆盖1/2",
        "毛覆盖全部"
    ],
    "下腹部": [
        "无毛",
        "中线少许毛",
        "中线毛呈条状",
        "中线毛呈带状",
        "中线毛呈倒V形状"
    ],
    "上背": [
        "无毛",
        "少许稀疏毛",
        "较多但仍分散",
        "完全覆盖，淡",
        "完全覆盖，浓"
    ],
    "下背部": [
        "无毛",
        "骶部一簇毛",
        "略向两侧伸展",
        "覆盖表面3/4",
        "完全覆盖"
    ],
    "上臂": [
        "无毛",
        "毛稀疏，未超过表面1/4",
        "超过1/4，但未完全覆盖",
        "完全覆盖，毛淡",
        "完全覆盖，毛浓"
    ],
    "大腿": [
        "无毛",
        "毛稀疏，未超过表面1/4",
        "超过1/4，但未完全覆盖",
        "完全覆盖，毛淡",
        "完全覆盖，毛浓"
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
        return "达到多毛参考标准"
    return "未达到多毛参考标准"


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
    # 黑白图版本：较高风险用深黑，较低风险用中灰，避免红绿配色
    if p >= threshold:
        return "较高风险倾向", "#111111"
    return "较低风险倾向", "#666666"


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

    # 固定随机种子，减少 Kernel SHAP 抽样近似导致的波动。
    np.random.seed(42)
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


def make_input_cache_key(X_one_row):
    """
    将当前输入转换成稳定缓存键。
    同一组输入重复点击“开始评估”时，直接复用上一次 SHAP 结果，
    避免特征贡献图反复重算导致排序和数值轻微变化。
    """
    vals = X_one_row.iloc[0].to_numpy(dtype=float)
    vals = np.round(vals, 6)
    return tuple(vals.tolist())


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
    colors = np.where(x_vals >= 0, "#111111", "#9E9E9E")

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
        textfont=dict(color=textcolor[::-1], size=16),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>贡献值：%{x:+.2f} 个百分点<extra></extra>",
    ))

    fig.update_layout(
        height=max(360, 50 * len(labels) + 100),
        margin=dict(l=360, r=100, t=20, b=50),
        font=dict(family="Microsoft YaHei, PingFang SC, Arial", size=20),
        yaxis=dict(
            title="",
            type="category",
            tickfont=dict(size=18),
            automargin=True
        ),
        xaxis=dict(
            title="特征贡献值（百分点）",
            zeroline=True,
            zerolinewidth=1.2,
            zerolinecolor="#222222",
            showgrid=True,
            gridcolor="#D9D9D9",
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
        line_color="#222222",
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
st.title("多囊卵巢综合征无创风险评估")
st.caption("基于无创临床指标和问卷条目评估个体化 PCOS 风险。")

init_session_state()


# ========================
# 输入区域
# 不使用 st.form，保证多毛评分解释可以实时更新
# ========================
with st.container():

    st.markdown("### 一、基本信息与月经情况")

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        age = text_number_input(
            FEATURE_META["Age"]["label"],
            "Age_input",
            10,
            60,
            is_int=True,
            digits=0
        )

        age_menarche = st.selectbox(
            "月经初潮年龄",
            options=list(OPTION_MAP["Age at menarche"].keys()),
            key="Age at menarche_input"
        )

        hair_loss = st.selectbox(
            "是否存在头顶部毛发稀疏或明显脱发？",
            options=list(OPTION_MAP["Hair loss"].keys()),
            key="Hair loss_input"
        )

    with col_right:
        menstrual_cycle = st.selectbox(
            "月经周期情况",
            options=list(OPTION_MAP["Menstrual cycle regularity"].keys()),
            key="Menstrual cycle regularity_input"
        )

    st.markdown("### 二、体格测量及身体成分")

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        height_cm = text_number_input(
            "身高（cm）",
            "height_cm_input",
            120.0,
            220.0,
            is_int=False,
            digits=1
        )

        weight_kg = text_number_input(
            "体重（kg）",
            "weight_kg_input",
            25.0,
            200.0,
            is_int=False,
            digits=1
        )

        bmi_value = weight_kg / ((height_cm / 100.0) ** 2)

        st.markdown(
            f"""
            <div style="
                background-color:#F5F5F5;
                padding:12px 16px;
                border-radius:8px;
                border:1px solid #222222;
                font-size:16px;
                color:#111111;
                margin-bottom:16px;">
                <b>BMI：</b>{bmi_value:.2f} kg/m<sup>2</sup>
            </div>
            """,
            unsafe_allow_html=True
        )

        waist = text_number_input(
            FEATURE_META["Waist circumference"]["label"],
            "Waist circumference_input",
            40.0,
            180.0,
            is_int=False,
            digits=1
        )

        sbp = text_number_input(
            FEATURE_META["Systolic blood pressure"]["label"],
            "Systolic blood pressure_input",
            70,
            250,
            is_int=True,
            digits=0
        )

        skeletal_muscle = text_number_input(
            FEATURE_META["Skeletal muscle mass"]["label"],
            "Skeletal muscle mass_input",
            5.0,
            80.0,
            is_int=False,
            digits=1
        )

    with col_right:
        neck = text_number_input(
            FEATURE_META["Neck circumference"]["label"],
            "Neck circumference_input",
            20.0,
            60.0,
            is_int=False,
            digits=1
        )

        hip = text_number_input(
            FEATURE_META["Hip circumference"]["label"],
            "Hip circumference_input",
            50.0,
            200.0,
            is_int=False,
            digits=1
        )

        dbp = text_number_input(
            FEATURE_META["Diastolic blood pressure"]["label"],
            "Diastolic blood pressure_input",
            40,
            150,
            is_int=True,
            digits=0
        )

        body_fat = text_number_input(
            FEATURE_META["Percent body fat"]["label"],
            "Percent body fat_input",
            3.0,
            80.0,
            is_int=False,
            digits=1
        )

    st.markdown("### 三、症状量表评分")

    with st.expander("痤疮评分", expanded=False):
        st.caption("请选择各区域最严重的皮损情况，系统将自动计算痤疮评分。")

        acne_selections = {}

        col1, col2 = st.columns(2, gap="medium")

        for i, region in enumerate(ACNE_REGIONS):
            target_col = col1 if i % 2 == 0 else col2

            with target_col:
                acne_options = list(ACNE_OPTIONS.keys())
                acne_default_choice = ACNE_DEFAULT_SELECTIONS.get(region, "无皮损")
                acne_default_index = acne_options.index(acne_default_choice)

                acne_selections[region] = st.selectbox(
                    f"{region}最严重皮损情况",
                    options=acne_options,
                    index=acne_default_index,
                    key=f"acne_select_{region}"
                )

        acne_total = calculate_acne_score(acne_selections)

        st.info(
            f"痤疮评分：{acne_total:.0f} / 44；"
            f"分级：{classify_acne_score(acne_total)}"
        )

    with st.expander("多毛评分", expanded=False):
        st.caption("请选择各部位毛发生长情况，系统将自动计算多毛评分。")

        hirsutism_scores = {}

        col1, col2 = st.columns(2, gap="medium")

        for i, (item, descriptions) in enumerate(HIRSUTISM_ITEMS.items()):
            target_col = col1 if i % 2 == 0 else col2

            with target_col:
                st.markdown(f"**{item}**")

                score = st.radio(
                    label=f"{item}评分",
                    options=[0, 1, 2, 3, 4],
                    index=HIRSUTISM_DEFAULT_SCORES.get(item, 0),
                    horizontal=True,
                    key=f"hirsutism_radio_{item}",
                    label_visibility="collapsed"
                )

                st.caption(f"当前选择：{score}分，{descriptions[score]}")
                hirsutism_scores[item] = int(score)

        hirsutism_total = calculate_hirsutism_score(hirsutism_scores)
        hirsutism_level = classify_hirsutism_score(hirsutism_scores)

        st.info(
            f"多毛评分：{hirsutism_total:.0f} / 36；"
            f"{hirsutism_level}"
        )

    if st.button("开始评估", type="primary", use_container_width=True):
        st.session_state["submitted"] = True

    submitted = st.session_state["submitted"]


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
        <div style="
            border:1.5px solid #111111;
            border-radius:8px;
            padding:14px 18px;
            margin-top:16px;
            margin-bottom:10px;
            background:#FFFFFF;
            font-family:Microsoft YaHei, PingFang SC, Arial;
            font-size:23px;">
          <b>PCOS 风险评估结果：{p1 * 100:.2f}%</b>
          <span class="badge" style="background:{pred_color};">{pred_label}</span>
          <div style="font-size:17px; margin-top:8px;">
            本结果用于 PCOS 风险识别和初步筛查参考，不作为临床诊断结论。
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("查看评估信息摘要"):
        input_summary = pd.DataFrame({
            "指标": [get_display_name(k) for k in MODEL_FEATURES],
            "取值": [format_feature_value(k, form_values[k]) for k in MODEL_FEATURES]
        })

        st.dataframe(
            input_summary,
            use_container_width=True,
            hide_index=True
        )

    if background_df is None:
        st.info(
            "预测结果已生成。若需显示局部特征贡献，请将 shap_background.csv、"
            "shap_background.xlsx 或 shap_background.xls 放入本应用同一文件夹。"
        )

    else:
        input_cache_key = make_input_cache_key(X)

        cached_shap = st.session_state.get("cached_shap_result", None)

        if cached_shap is not None and cached_shap.get("input_key") == input_cache_key:
            shap_vals = cached_shap.get("shap_vals")
        else:
            with st.spinner("正在计算局部特征贡献..."):
                shap_vals, _ = compute_kernel_shap_probability(X)

            if shap_vals is not None:
                st.session_state["cached_shap_result"] = {
                    "input_key": input_cache_key,
                    "shap_vals": shap_vals
                }

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