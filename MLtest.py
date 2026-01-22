# --- ขั้นตอนที่ 0: ติดตั้ง Library ที่จำเป็น ---
# pip install pandas scikit-learn matplotlib seaborn wordcloud nltk tensorflow
 
# ----------------------------------------
# --- import library ทั่วไป ---
import pandas as pd
import re
import sys
import warnings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import numpy as np
 
# --- Import NLTK (สำหรับภาษาอังกฤษ) ---
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
 
# --- Import libraries สำหรับ Deep Learning (TensorFlow/Keras) ---
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Embedding, LSTM, Bidirectional,
    GlobalMaxPooling1D, Conv1D, Dropout, SpatialDropout1D, GRU
)
from tensorflow.keras.optimizers import Adam
 
# --- Import สำหรับการประเมินผล ---
from sklearn.metrics import classification_report, confusion_matrix
 
# --- ดาวน์โหลดเครื่องมือของ NLTK (ทำครั้งแรกครั้งเดียว) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("--- กำลังดาวน์โหลด 'punkt' (สำหรับ word_tokenize) ---")
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    # ไม่ใช่ทุกสภาพแวดล้อมจะมี 'punkt_tab' แต่ให้พยายามดาวน์โหลดถ้าไม่มี
    try:
        print("--- กำลังดาวน์โหลด 'punkt_tab' ---")
        nltk.download('punkt_tab')
    except Exception:
        pass
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("--- กำลังดาวน์โหลด 'stopwords' (สำหรับตัดคำ) ---")
    nltk.download('stopwords')
 
# ปิดการแสดงผลคำเตือนที่ไม่จำเป็น
warnings.filterwarnings('ignore')
 
 
# ==============================================================================
# ส่วนที่ 1: การเตรียมข้อมูล (DATA PREPARATION)
# ==============================================================================
 
# --- 1. โหลดข้อมูลเข้าสู่ระบบด้วย pandas ---
print("--- 1. กำลังโหลดข้อมูลจาก CSV ---")
FPath = "D:\งาน\MLPro/Amazon_Unlocked_Mobile.csv"  # <--- แก้ไข Path นี้ถ้าจำเป็น
 
try:
    df_raw = pd.read_csv(FPath)
    df = df_raw[['Reviews', 'Rating']].copy()
    df = df.dropna(subset=['Reviews'])
    df = df.reset_index(drop=True)
    print("--- 1. โหลดข้อมูลดิบสำเร็จ ---")
except FileNotFoundError:
    print(f"Error: ไม่พบไฟล์ Dataset ที่ Path: {FPath}")
    sys.exit()
except Exception as e:
    print(f"เกิดข้อผิดพลาดอื่นในการโหลดข้อมูล: {e}")
    sys.exit()
 
# --- 6. การเลือก/สร้าง Label ที่เหมาะสม ---
def map_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    return None
 
df['sentiment'] = df['Rating'].apply(map_sentiment)
df = df.dropna(subset=['sentiment'])
df = df.reset_index(drop=True)
 
# --- 2, 3, 4, 5. การเตรียมและทำความสะอาดข้อมูล (Pipeline) [ฉบับภาษาอังกฤษ] ---
print("\n--- 2,3,4,5. กำลังทำความสะอาดข้อมูล (ภาษาอังกฤษ)... ---")
stop_words_eng = set(stopwords.words('english'))
 
def clean_text_pipeline_eng(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [
        word for word in tokens
        if word.isalpha() and len(word) > 1 and word not in stop_words_eng
    ]
    return " ".join(cleaned_tokens)
 
df['text_clean'] = df['Reviews'].apply(clean_text_pipeline_eng)
 
print("--- 2,3,4,5. ทำความสะอาดข้อมูลเสร็จสิ้น ---")
print(df[['text_clean', 'sentiment']].head())
 
# --- 7. ตรวจสอบความสมดุล (Imbalance) ของข้อมูล ---
print("\n--- 7. ตรวจสอบความสมดุลของข้อมูล ---")
print(df['sentiment'].value_counts())
 
 
# ==============================================================================
# ส่วนที่ 2: การวิเคราะห์ข้อมูลเบื้องต้น (EDA)
# ==============================================================================
 
print("\n\n--- เริ่มต้นการทำ Exploratory Data Analysis (EDA) ---")
sns.set(style="whitegrid")
print("--- ตั้งค่า Font (Default) สำหรับกราฟสำเร็จ ---")
 
 
# --- EDA 2.1: กราฟการกระจายของคลาส (Class Distribution) ---
print("\n--- EDA 2.1: แสดงกราฟการกระจายของคลาส ---")
plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment', data=df, palette=['#34A853', '#EA4335'])
plt.title('Distribution of Sentiments (Positive vs Negative)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
 
# --- EDA 2.2: กราฟวิเคราะห์ความยาวของข้อความ ---
print("\n--- EDA 2.2: วิเคราะห์และแสดงกราฟความยาวของข้อความ ---")
df['word_count'] = df['text_clean'].apply(lambda x: len(x.split()))
print("Word Count Statistics:")
print(df['word_count'].describe())
 
plt.figure(figsize=(12, 6))
sns.histplot(df['word_count'], bins=50, kde=True)
plt.title('Distribution of Word Count in Reviews (Cleaned)')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.xlim(0, 150)
plt.show()
 
# -------------------------------------------------------------------------------
# --- EDA 3.1: Word Clouds (แสดงคำที่พบบ่อย) [ฉบับภาษาอังกฤษ V2] ---
# -------------------------------------------------------------------------------
print("\n--- EDA 3.1: กำลังสร้าง Word Clouds (แบบมีสีและแนวตั้ง)... ---")
text_positive = " ".join(review for review in df[df['sentiment'] == 'positive']['text_clean'])
text_negative = " ".join(review for review in df[df['sentiment'] == 'negative']['text_clean'])
 
try:
    wordcloud_pos = WordCloud(
        width=1200, height=600,
        background_color='white',
        colormap='viridis',
        prefer_horizontal=0.6,
        min_font_size=10,
        max_words=200,
        random_state=42
    ).generate(text_positive)
 
    wordcloud_neg = WordCloud(
        width=1200, height=600,
        background_color='white',
        colormap='plasma',
        prefer_horizontal=0.6,
        min_font_size=10,
        max_words=200,
        random_state=42
    ).generate(text_negative)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 12))
    ax1.imshow(wordcloud_pos, interpolation='bilinear')
    ax1.set_title('Word Cloud - Positive Reviews', fontsize=24)
    ax1.axis('off')
    ax2.imshow(wordcloud_neg, interpolation='bilinear')
    ax2.set_title('Word Cloud - Negative Reviews', fontsize=24)
    ax2.axis('off')
    plt.show()
    print("--- สร้าง Word Clouds สำเร็จ ---")
except Exception as e:
    print(f"--- เกิดข้อผิดพลาดในการสร้าง Word Cloud: {e} ---")
 
 
# -------------------------------------------------------------------------------
# --- EDA 3.2: กราฟความถี่ของคำ (Word Frequency Graph) ---
# -------------------------------------------------------------------------------
print("\n--- EDA 3.2: กำลังสร้างกราฟความถี่ของคำ Top 20 ---")
 
def get_top_n_words(corpus, n=None):
    all_words = " ".join(corpus).split()
    word_counts = Counter(all_words)
    top_n_words = word_counts.most_common(n)
    return top_n_words
 
top_words_positive = get_top_n_words(df[df['sentiment'] == 'positive']['text_clean'], n=20)
top_words_negative = get_top_n_words(df[df['sentiment'] == 'negative']['text_clean'], n=20)
 
df_top_pos = pd.DataFrame(top_words_positive, columns=['word', 'count'])
df_top_neg = pd.DataFrame(top_words_negative, columns=['word', 'count'])
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
sns.barplot(x='count', y='word', data=df_top_pos, ax=ax1, palette='Greens_r')
ax1.set_title('Top 20 Words in Positive Reviews')
ax1.set_xlabel('Count')
ax1.set_ylabel('Word')
sns.barplot(x='count', y='word', data=df_top_neg, ax=ax2, palette='Reds_r')
ax2.set_title('Top 20 Words in Negative Reviews')
ax2.set_xlabel('Count')
ax2.set_ylabel('Word')
plt.tight_layout()
plt.show()
 
 
# ==============================================================================
# ส่วนที่ 3: การแบ่งข้อมูล (DATA SPLITTING)
# ==============================================================================
 
print("\n\n--- 3. กำลังแบ่งข้อมูล Train/Test ---")
X = df['text_clean']
y = df['sentiment']
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
 
print(f"ข้อมูล Train (X_train): {len(X_train)} แถว")
print(f"ข้อมูล Test (X_test): {len(X_test)} แถว")
 
 
# ==============================================================================
# ส่วนที่ 4: การเตรียมข้อมูลสำหรับ DEEP LEARNING
# ==============================================================================
 
# --- 4.1 แปลง Label (String) เป็นตัวเลข (Binary) ---
print("\n--- 4.1 กำลังแปลง Labels เป็น Binary (0, 1) ---")
y_train_binary = y_train.map({'positive': 1, 'negative': 0}).astype(int)
y_test_binary = y_test.map({'positive': 1, 'negative': 0}).astype(int)
 
# --- 4.2 ตั้งค่า Hyperparameters ---
VOCAB_SIZE = 10000  # จำนวนคำสูงสุด
MAX_LEN = 150       # ความยาวสูงสุดของรีวิว
EMBEDDING_DIM = 128 # ขนาดของเวกเตอร์คำ
 
# --- 4.3 สร้าง Tokenizer และแปลงข้อความเป็น Sequences ---
print("--- 4.3 กำลังแปลงข้อความเป็น Sequences ---")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
 
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
 
# --- 4.4 ทำ Padding ให้ทุก Sequence มีความยาวเท่ากัน ---
print("--- 4.4 กำลังทำ Padding ข้อมูล ---")
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')
 
print(f"ขนาดข้อมูล Train (Padded): {X_train_pad.shape}")
print(f"ขนาดข้อมูล Test (Padded): {X_test_pad.shape}")
 
 
# ==============================================================================
# ส่วนที่ 5: การพัฒนาโมเดล 1 (Bi-LSTM)
# ==============================================================================
 
print("\n\n--- 5. การพัฒนาโมเดลที่ 1: Bidirectional LSTM ---")
 
model_bilstm = Sequential([
    Embedding(input_dim=VOCAB_SIZE,
              output_dim=EMBEDDING_DIM,
              input_length=MAX_LEN),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(units=64, return_sequences=False)),
    Dropout(0.3),
    Dense(units=32, activation='relu'),
    Dense(units=1, activation='sigmoid')
])
 
# --- แก้ไขที่นี่: build ก่อนเรียก summary ---
model_bilstm.build(input_shape=(None, MAX_LEN))
 
print("--- Architecture ของ Bi-LSTM Model ---")
model_bilstm.summary()
 
model_bilstm.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
 
# --- ฝึกสอนโมเดล (Training) ---
print("\n--- 5.1 กำลังฝึกสอนโมเดล Bi-LSTM (อาจใช้เวลาสักครู่)... ---")
EPOCHS = 5
BATCH_SIZE = 64
 
history_bilstm = model_bilstm.fit(
    X_train_pad,
    y_train_binary,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test_pad, y_test_binary),
    verbose=1
)
 
# --- 5.2 ประเมินผลโมเดล Bi-LSTM ---
print("\n--- 5.2 การประเมินผลโมเดล Bi-LSTM ด้วย Test Dataset ---")
loss, accuracy = model_bilstm.evaluate(X_test_pad, y_test_binary, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
 
y_pred_probs_bilstm = model_bilstm.predict(X_test_pad)
y_pred_classes_bilstm = (y_pred_probs_bilstm > 0.5).astype(int).flatten()
 
print("\n--- Classification Report (Bi-LSTM) ---")
print(classification_report(y_test_binary, y_pred_classes_bilstm, target_names=['Negative', 'Positive']))
 
 
# ==============================================================================
# ส่วนที่ 6: การพัฒนาโมเดล 2 (CNN)
# ==============================================================================
 
print("\n\n--- 6. การพัฒนาโมเดลที่ 2: CNN (เพื่อเปรียบเทียบ) ---")
 
model_cnn = Sequential([
    Embedding(input_dim=VOCAB_SIZE,
              output_dim=EMBEDDING_DIM,
              input_length=MAX_LEN),
    Dropout(0.2),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(units=64, activation='relu'),
    Dropout(0.3),
    Dense(units=1, activation='sigmoid')
])
 
# --- แก้ไขที่นี่: build ก่อนเรียก summary ---
model_cnn.build(input_shape=(None, MAX_LEN))
 
print("--- Architecture ของ CNN Model ---")
model_cnn.summary()
 
model_cnn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
 
# --- ฝึกสอนโมเดล (Training) ---
print("\n--- 6.1 กำลังฝึกสอนโมเดล CNN (อาจใช้เวลาสักครู่)... ---")
history_cnn = model_cnn.fit(
    X_train_pad,
    y_train_binary,
    epochs=EPOCHS,  # ใช้จำนวน Epoch เท่ากันเพื่อเปรียบเทียบ
    batch_size=BATCH_SIZE,
    validation_data=(X_test_pad, y_test_binary),
    verbose=1
)
 
# --- 6.2 ประเมินผลโมเดล CNN ---
print("\n--- 6.2 การประเมินผลโมเดล CNN ด้วย Test Dataset ---")
loss_cnn, acc_cnn = model_cnn.evaluate(X_test_pad, y_test_binary, verbose=0)
print(f"Test Loss: {loss_cnn:.4f}")
print(f"Test Accuracy: {acc_cnn:.4f}")
 
y_pred_probs_cnn = model_cnn.predict(X_test_pad)
y_pred_classes_cnn = (y_pred_probs_cnn > 0.5).astype(int).flatten()
 
print("\n--- Classification Report (CNN) ---")
print(classification_report(y_test_binary, y_pred_classes_cnn, target_names=['Negative', 'Positive']))
 
print("\n🎉🎉🎉 สิ้นสุดกระบวนการทั้งหมด! 🎉🎉🎉")
 