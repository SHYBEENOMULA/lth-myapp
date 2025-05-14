import streamlit as st
from cnocr import CnOcr
from PIL import Image
import re
import time
from sparkai.llm.llm import ChatSparkLLM
from sparkai.core.messages import ChatMessage

# === 页面配置，必须最先调用 ===
st.set_page_config(page_title="智能食品成分分析", layout="wide")

# 初始化 OCR 时添加离线配置
ocr = CnOcr(
    model_name='densenet_lite_136-fc',
    model_dir='./models',  # 本地模型目录
    root='./'  # 强制禁用自动下载
)

# === 自定义CSS样式 ===
st.markdown(
    """
    <style>
        .block-container { padding: 2rem 1rem; background-color: #f8f9fa; }
        .header { color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: .5rem; margin-bottom: 1rem; }
        .result-box { background-color: #fff; border-radius: 10px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,.1); margin-top: 1rem; white-space: pre-wrap; }
        .warning { color: #ff6b6b; font-weight: bold; }
        .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
        .stButton>button:hover { background-color: #45a049; }
        .stTextInput>div>div>input { border: 1px solid #dee2e6; }
    </style>
    """, unsafe_allow_html=True
)

# === 初始化OCR和LLM ===
ocr = CnOcr()
SPARKAI_CONFIG = {
    "url": 'wss://spark-api.xf-yun.com/v3.5/chat',
    "app_id": 'fa4d9cbe',
    "api_key": '42bf5b15021df8c8972352c5104a5add',
    "api_secret": 'MjFiMjJmOTE3ZDg4MjA5NjQwMjNiYzE3',
    "domain": 'generalv3.5'
}

# 常见添加剂关键词，用于校验
ADDITIVE_KEYWORDS = [
    '酸', '盐', '糖', '脂肪', '油', '香精', '防腐剂', '色素', '磷酸', '乳化剂', '抗氧化剂'
]

# === 功能函数 ===
def recognize_image(file):
    img = Image.open(file)
    res = ocr.ocr(img)
    text_lines = [''.join(item['text']) for item in res]
    full_text = '\n'.join([l.split(':',1)[-1].strip() for l in text_lines])
    chunks = re.split(r'[\,;，；:\n]', full_text)
    phrases = [c.strip() for c in chunks if c.strip()]
    return full_text, phrases


def clean_text(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = re.sub(r'[*#`-]+', '', line)
        cleaned.append(line.strip())
    return '\n'.join([l for l in cleaned if l])


def analyze_ingredients(phrases):
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_CONFIG['url'],
        spark_app_id=SPARKAI_CONFIG['app_id'],
        spark_api_key=SPARKAI_CONFIG['api_key'],
        spark_api_secret=SPARKAI_CONFIG['api_secret'],
        spark_llm_domain=SPARKAI_CONFIG['domain'],
        streaming=False
    )
    prompt = (
        "请分析以下食品成分对儿童健康的影响，用中文回答，仅使用纯文本格式，不包含任何Markdown语法或符号：\n"
        f"成分列表：{', '.join(phrases)}\n"
        "要求：\n"
        "1. 按危险等级分类（高/中/低）\n"
        "2. 给出简明建议\n"
        "3. 使用通俗易懂的语言\n"
        "4. 包含每日建议摄入量（如适用）"
    )
    msgs = [ChatMessage(role="user", content=prompt)]
    result = spark.generate([msgs])
    return clean_text(result.generations[0][0].text)

# === 页面布局 ===
st.title("智能食品成分分析")

st.markdown(
    '<div class="header">上传食品包装图片，获取专业育儿食品安全建议</div>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader("上传食品包装图片：", type=['png','jpg','jpeg'])

if uploaded:
    cols = st.columns([2,3])
    full_text, phrases = recognize_image(uploaded)
    with cols[0]:
        st.image(uploaded, caption="图片预览", use_column_width=True)
    with cols[1]:
        st.subheader("识别结果")
        with st.expander("查看完整文本"): st.text(full_text)

    select_all = st.checkbox("一键全选所有成分")
    selected = phrases if select_all else st.multiselect("选择分析成分：", options=phrases, default=phrases[:3])

    if st.button("开始健康分析"):
        if not selected:
            st.warning("请至少选择一个成分进行分析")
        else:
                        # 校验所选成分是否属于常见添加剂
            invalid = [p for p in selected if not any(kw in p for kw in ADDITIVE_KEYWORDS)]
            if invalid:
                st.error("以下所选内容不属于常见添加剂，请重新选择：" + ", ".join(invalid))
            else:
                with st.spinner("分析进行中..."):
                    res = analyze_ingredients(selected)
                st.markdown(
                    f'<div class="result-box">{res}</div>',
                    unsafe_allow_html=True
                )
else:
    st.info("使用说明：上传图片后，系统将自动识别并切分成分块，可选中成分后获取健康分析建议。请确保图片清晰。可一键全选或手动选择成分。添加剂分析需至少包含酸、盐、糖、脂肪等关键词。")

st.caption("基于星火大模型与OCR技术，结果仅供参考，具体建议请咨询专业营养师")
