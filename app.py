
import os
import re
import joblib
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    from pyvi import ViTokenizer
except Exception:
    ViTokenizer = None

try:
    import emoji
except Exception:
    emoji = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Vietnamese Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

SVM_MODEL_PATH = "models/svm_model.pkl"
BILSTM_MODEL_PATH = "models/bilstm_model.pt"
PHOBERT_MODEL_PATH = "models/phobert_model"

LABEL_MAP = {0: "NEG", 1: "NEU", 2: "POS"}
LABEL_ICON = {"NEG": "😠", "NEU": "😐", "POS": "😊"}
LABEL_COLOR = {"NEG": "#ef4444", "NEU": "#f59e0b", "POS": "#22c55e"}


# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        max-width: 1180px;
    }

    .hero-card {
        padding: 30px;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.25);
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-desc {
        font-size: 16px;
        color: #dbeafe;
        line-height: 1.6;
    }

    .panel {
        background: white;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        margin-bottom: 20px;
    }

    .result-card {
        background: white;
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        text-align: center;
        min-height: 155px;
    }

    .model-name {
        font-size: 18px;
        font-weight: 700;
        color: #334155;
        margin-bottom: 10px;
    }

    .label {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .conf {
        font-size: 15px;
        color: #64748b;
    }

    .status {
        padding: 10px 12px;
        background: #1e293b;
        border-radius: 12px;
        margin-bottom: 8px;
        color: white;
        font-size: 14px;
    }

    div[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    div[data-testid="stSidebar"] * {
        color: white;
    }

    .stButton > button {
        border-radius: 12px;
        height: 46px;
        font-weight: 700;
        background: #2563eb;
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        color: white;
        border: none;
    }

    .stDownloadButton > button {
        border-radius: 12px;
        height: 46px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PREPROCESSING
# =========================================================

ABBREVIATION_MAP = {
    "ko": "không", "k": "không", "kh": "không", "kg": "không", "khg": "không",
    "hok": "không", "hong": "không", "hông": "không", "hem": "không", "0": "không",
    "dc": "được", "đc": "được", "đx": "được",
    "okie": "ok", "oke": "ok", "okela": "ok",
    "sp": "sản phẩm", "mn": "mọi người", "mng": "mọi người",
    "mk": "mình", "mik": "mình", "tui": "tôi",
    "vs": "với", "v": "vậy", "r": "rồi", "ùi": "rồi",
    "lun": "luôn", "ntn": "như thế nào", "j": "gì",
    "z": "vậy", "zậy": "vậy", "wa": "quá", "wá": "quá", "qá": "quá",
    "rat": "rất", "rât": "rất",
    "bt": "bình thường", "bth": "bình thường", "bthg": "bình thường",
    "sd": "sử dụng", "xài": "dùng",
    "ngonvl": "rất ngon", "đỉnhvl": "rất tốt", "xịn": "tốt",
    "chánvl": "rất chán", "tệvl": "rất tệ", "vl": "rất", "vcl": "rất", "vc": "rất",
    "cute": "dễ thương", "iu": "yêu", "iuu": "yêu",
}

POSITIVE_EMOTICONS = {":)", ":-)", ":d", ":D", "=)", "^^", "^_^", ":3", ";)", "<3",
                      "😍", "🥰", "😘", "😊", "😄", "😁", "😆", "👍", "👌", "❤️", "❤", "💯", "🌟", "⭐"}
NEGATIVE_EMOTICONS = {":(", ":-(", ":'(", ":/", "😢", "😭", "😞", "😔", "😡", "🤬", "👎", "💔", "😒"}

POSITIVE_PHRASES = [
    "miễn chê", "khỏi chê", "không chê vào đâu được",
    "không có gì để chê", "không tệ", "không tồi",
    "chưa bao giờ thất vọng", "không thất vọng"
]
NEGATIVE_PHRASES = [
    "không nên mua", "đừng mua", "không đáng tiền", "không hài lòng",
    "không ưng", "không ổn", "không dùng được", "chưa dùng đã hỏng"
]
NEGATION_WORDS = {"không", "chưa", "chẳng", "chả", "đừng", "khỏi"}
PUNCTUATIONS = {".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}"}


def normalize_unicode(text):
    return unicodedata.normalize("NFC", str(text))


def reduce_repeated_chars(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize_emojis_and_icons(text):
    for emo in sorted(POSITIVE_EMOTICONS, key=len, reverse=True):
        text = text.replace(emo, " EMO_POS ")
    for emo in sorted(NEGATIVE_EMOTICONS, key=len, reverse=True):
        text = text.replace(emo, " EMO_NEG ")

    if emoji is None:
        return text

    chars = []
    for ch in text:
        if ch in emoji.EMOJI_DATA:
            if ch in POSITIVE_EMOTICONS:
                chars.append(" EMO_POS ")
            elif ch in NEGATIVE_EMOTICONS:
                chars.append(" EMO_NEG ")
            else:
                chars.append(" EMO_UNK ")
        else:
            chars.append(ch)
    return "".join(chars)


def normalize_abbreviations(text):
    return " ".join([ABBREVIATION_MAP.get(tok.strip(), tok.strip()) for tok in text.split()])


def basic_clean_text(text):
    text = normalize_unicode(text).lower()
    text = normalize_emojis_and_icons(text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\b\d{9,12}\b", " PHONE ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = reduce_repeated_chars(text)
    text = re.sub(r"([!?.,;:()\[\]{}])", r" \1 ", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ_!?.,;:()\[\]{}\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_abbreviations(text)
    return re.sub(r"\s+", " ", text).strip()


def replace_special_phrases(text):
    for phrase in sorted(POSITIVE_PHRASES, key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", " POS_PHRASE ", text)
    for phrase in sorted(NEGATIVE_PHRASES, key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", " NEG_PHRASE ", text)
    return re.sub(r"\s+", " ", text).strip()


def handle_negation(text, window=2):
    text = replace_special_phrases(text)
    tokens = text.split()
    result = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        if tok in NEGATION_WORDS:
            phrase = [tok]
            j = i + 1
            steps = 0
            while j < len(tokens) and steps < window:
                if tokens[j] in PUNCTUATIONS:
                    break
                phrase.append(tokens[j])
                j += 1
                steps += 1
            if len(phrase) > 1:
                result.append("_".join(phrase))
                i = j
            else:
                result.append(tok)
                i += 1
        else:
            result.append(tok)
            i += 1

    return " ".join(result)


def word_segment(text):
    if ViTokenizer is None:
        return text
    return ViTokenizer.tokenize(text)


def preprocess_text(text):
    text = basic_clean_text(text)
    text = handle_negation(text, window=2)
    text = word_segment(text)
    return text


# =========================================================
# BILSTM
# =========================================================

if nn is not None:
    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, num_classes=3, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, input_ids):
            embedded = self.dropout(self.embedding(input_ids))
            _, (hidden, _) = self.lstm(embedded)
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            final_hidden = self.dropout(final_hidden)
            return self.fc(final_hidden)


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_svm_model():
    if not os.path.exists(SVM_MODEL_PATH):
        return None
    return joblib.load(SVM_MODEL_PATH)

@st.cache_resource
def load_bilstm_model():
    if torch is None or nn is None or not os.path.exists(BILSTM_MODEL_PATH):
        return None, None, None

    checkpoint = torch.load(BILSTM_MODEL_PATH, map_location="cpu")
    vocab = checkpoint.get("vocab")
    max_len = checkpoint.get("max_len", 100)

    if vocab is None:
        return None, None, None

    model = BiLSTMClassifier(len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab, max_len

@st.cache_resource
def load_phobert_model():
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        return None, None
    if not os.path.exists(PHOBERT_MODEL_PATH):
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_PATH, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(PHOBERT_MODEL_PATH)
    model.eval()
    return tokenizer, model

# =========================================================
# PREDICT
# =========================================================
def text_to_ids(text, vocab, max_len=100):
    tokens = str(text).split()[:max_len]
    return [vocab.get(tok, vocab.get("<UNK>", 1)) for tok in tokens]

def predict_svm(text, model):
    if model is None:
        return None

    processed = preprocess_text(text)
    pred = model.predict([processed])[0]
    conf = None

    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function([processed])[0]
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            conf = float(np.max(probs))
        except Exception:
            pass

    return {"model": "SVM", "label": LABEL_MAP.get(int(pred), str(pred)), "confidence": conf, "processed_text": processed}

def predict_bilstm(text, model, vocab, max_len):
    if torch is None or model is None or vocab is None:
        return None

    processed = preprocess_text(text)
    ids = text_to_ids(processed, vocab, max_len)
    input_ids = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = float(probs[0, pred].item())

    return {"model": "BiLSTM", "label": LABEL_MAP.get(int(pred), str(pred)), "confidence": conf, "processed_text": processed}


def predict_phobert(text, tokenizer, model):
    if torch is None or tokenizer is None or model is None:
        return None

    processed = preprocess_text(text)
    encoding = tokenizer(processed, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = float(probs[0, pred].item())

    return {"model": "PhoBERT", "label": LABEL_MAP.get(int(pred), str(pred)), "confidence": conf, "processed_text": processed}


def render_result_card(result):
    label = result["label"]
    color = LABEL_COLOR.get(label, "#64748b")
    icon = LABEL_ICON.get(label, "")
    conf = result["confidence"]

    conf_html = "N/A" if conf is None else f"{conf:.4f}"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="model-name">{result['model']}</div>
            <div class="label" style="color:{color};">{icon} {label}</div>
            <div class="conf">Confidence: {conf_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGES
# =========================================================

def page_single_review(svm_model, bilstm_model, bilstm_vocab, bilstm_max_len, phobert_tokenizer, phobert_model):
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">💬 Vietnamese Sentiment Classifier</div>
            <div class="hero-desc">
                Phân tích cảm xúc đánh giá sản phẩm tiếng Việt với SVM, BiLSTM và PhoBERT.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns([1.25, 0.75])

    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        review = st.text_area(
            "Nhập nội dung review",
            height=190,
            placeholder="Ví dụ: Sản phẩm đẹp, giao hàng nhanh, đóng gói cẩn thận..."
        )
        selected_models = st.multiselect(
            "Chọn mô hình",
            ["SVM", "BiLSTM", "PhoBERT"],
            default=["SVM", "BiLSTM", "PhoBERT"]
        )
        run = st.button("Phân tích cảm xúc", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Gợi ý test")
        for ex in [
            "Sản phẩm rất tốt, giao hàng nhanh.",
            "Ko tốt lắm, pin tụt nhanh :(",
            "Dùng bình thường, không có gì nổi bật.",
            "Miễn chê luôn, rất đáng mua <3",
            "Không nên mua, chất lượng quá tệ."
        ]:
            st.code(ex, language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if not review.strip():
            st.warning("Vui lòng nhập review.")
            return

        result_list = []
        if "SVM" in selected_models:
            r = predict_svm(review, svm_model)
            if r: result_list.append(r)
        if "BiLSTM" in selected_models:
            r = predict_bilstm(review, bilstm_model, bilstm_vocab, bilstm_max_len)
            if r: result_list.append(r)
        if "PhoBERT" in selected_models:
            r = predict_phobert(review, phobert_tokenizer, phobert_model)
            if r: result_list.append(r)

        if not result_list:
            st.error("Không có mô hình nào được load. Hãy kiểm tra thư mục models/.")
            return

        st.subheader("Kết quả dự đoán")
        cols = st.columns(len(result_list))
        for col, result in zip(cols, result_list):
            with col:
                render_result_card(result)

        with st.expander("Văn bản sau tiền xử lý"):
            st.code(result_list[0]["processed_text"], language="text")


def page_csv_prediction(svm_model, bilstm_model, bilstm_vocab, bilstm_max_len, phobert_tokenizer, phobert_model):
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">📁 Batch Prediction</div>
            <div class="hero-desc">
                Upload file CSV chứa nhiều bình luận và xuất kết quả dự đoán. Confidence được lưu với 4 chữ số thập phân.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    c1, c2 = st.columns(2)
    with c1:
        model_choice = st.selectbox("Chọn mô hình", ["SVM", "BiLSTM", "PhoBERT"])
    with c2:
        max_rows = st.number_input("Số dòng tối đa dự đoán", min_value=1, value=50, step=10)

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Xem trước dữ liệu")
        st.dataframe(df_upload.head(10), use_container_width=True)

        text_col = st.selectbox("Chọn cột chứa review", df_upload.columns)
        run = st.button("Dự đoán file CSV", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run:
            df_run = df_upload.head(int(max_rows)).copy()
            output = []
            progress = st.progress(0)

            for idx, text in enumerate(df_run[text_col].astype(str)):
                if model_choice == "SVM":
                    result = predict_svm(text, svm_model)
                elif model_choice == "BiLSTM":
                    result = predict_bilstm(text, bilstm_model, bilstm_vocab, bilstm_max_len)
                else:
                    result = predict_phobert(text, phobert_tokenizer, phobert_model)

                if result is None:
                    output.append({
                        "review": text,
                        "predicted_label": "MODEL_NOT_FOUND",
                        "confidence": None
                    })
                else:
                    output.append({
                        "review": text,
                        "predicted_label": result["label"],
                        "confidence": round(result["confidence"], 4) if result["confidence"] is not None else None
                    })

                progress.progress((idx + 1) / len(df_run))

            result_df = pd.DataFrame(output)

            st.subheader("Kết quả")
            st.dataframe(result_df, use_container_width=True)

            label_counts = result_df["predicted_label"].value_counts()
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng review", len(result_df))
            col2.metric("Nhãn nhiều nhất", label_counts.index[0])
            col3.metric("Số lượng", int(label_counts.iloc[0]))

            st.subheader("Phân bố nhãn dự đoán")
            st.bar_chart(label_counts)

            csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Tải kết quả CSV",
                data=csv_data,
                file_name="sentiment_prediction_result.csv",
                mime="text/csv",
                use_container_width=True
            )


def page_about():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">ℹ️ About</div>
            <div class="hero-desc">
                Ứng dụng NLP trong phân tích cảm xúc đánh giá sản phẩm tiếng Việt.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(
        """
        ### Chức năng chính

        - Dự đoán cảm xúc một review đơn lẻ.
        - Dự đoán cảm xúc hàng loạt từ file CSV.
        - Hỗ trợ 3 mô hình: **SVM**, **BiLSTM**, **PhoBERT**.

        ### Cấu trúc model cần có

        ```text
        models/
        ├── svm_model.pkl
        ├── bilstm_model.pt
        └── phobert_model/
        ```

        ### Chạy ứng dụng

        ```bash
        streamlit run app.py
        ```
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# MAIN
# =========================================================

def main():
    svm_model = load_svm_model()
    bilstm_model, bilstm_vocab, bilstm_max_len = load_bilstm_model()
    phobert_tokenizer, phobert_model = load_phobert_model()

    st.sidebar.markdown("## 💬 Sentiment App")
    st.sidebar.caption("Product Review Classification")

    page = st.sidebar.radio(
        "Menu",
        ["Predict Single Review", "Predict CSV File", "About"]
    )

    st.sidebar.divider()
    st.sidebar.markdown("### Model Status")

    st.sidebar.markdown(f"<div class='status'>SVM: {'✅ Loaded' if svm_model else '❌ Not found'}</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='status'>BiLSTM: {'✅ Loaded' if bilstm_model else '❌ Not found'}</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='status'>PhoBERT: {'✅ Loaded' if phobert_model else '❌ Not found'}</div>", unsafe_allow_html=True)

    st.sidebar.divider()
    st.sidebar.caption("Confidence khi xuất CSV được làm tròn 4 chữ số thập phân.")

    if page == "Predict Single Review":
        page_single_review(svm_model, bilstm_model, bilstm_vocab, bilstm_max_len, phobert_tokenizer, phobert_model)
    elif page == "Predict CSV File":
        page_csv_prediction(svm_model, bilstm_model, bilstm_vocab, bilstm_max_len, phobert_tokenizer, phobert_model)
    else:
        page_about()


if __name__ == "__main__":
    main()
