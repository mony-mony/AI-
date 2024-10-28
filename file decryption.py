from gtts import gTTS
from playsound import playsound
import time, os
import speech_recognition as sr
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from itertools import cycle  # 색상을 순차적으로 적용하기 위한 모듈

AI_clear = 1
SIMILARITY_THRESHOLD = 0.9999999  # 목소리 비슷한 정도 임계값 설정
CSV_FILE = 'audio_password.csv'  # CSV 파일 경로

similarity_scores_per_folder = {}  # 각 폴더별 유사도 값을 저장할 딕셔너리

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('text_password.csv', encoding='cp949')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_features(y, sr, n_mfcc=192):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def compare_audio_features(features1, features2):
    return 1 - cosine(features1, features2)

def answer(input_text, audio_path, filename, file_address, folder_name):
    model = cached_model()
    df = get_dataset()
    global AI_clear

    if "돌아가" in input_text or '모르겠어' in input_text or '몰라' in input_text:
        speak("다시 메인으로 돌아가겠습니다.")
        AI_clear = 3
        return
    
    speak("임베딩 시작")

    embedding = model.encode(input_text)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    y, sr = load_audio(audio_path)
    input_features = extract_features(y, sr, n_mfcc=192)

    audio_data_df = pd.read_csv(CSV_FILE, converters={'features': eval})
    results = []

    for index, row in audio_data_df.iterrows():
        audio_features = np.array(row['features'])
        similarity = compare_audio_features(input_features, audio_features)
        results.append((row['file_path'], similarity))

    most_similar = max(results, key=lambda x: x[1])

    # 폴더별로 유사도 값을 저장
    if folder_name not in similarity_scores_per_folder:
        similarity_scores_per_folder[folder_name] = []
    
    similarity_scores_per_folder[folder_name].append(most_similar[1])  # 유사도 값을 폴더별로 저장

    print(f"Most similar file: {most_similar[0]} with similarity: {most_similar[1]}")

    df_new = pd.DataFrame({'password': [input_text], 'similarity': [most_similar[1]]})

    if most_similar[1] >= SIMILARITY_THRESHOLD:
        speak("암호를 확인했습니다.")
        csv_file = 'result.csv'
        if os.path.exists(csv_file):
            df_existing = pd.read_csv(csv_file, converters={'features': eval})
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(csv_file, index=False)
    else:
        speak("사용자의 음성이 다릅니다.")
        csv_file = 'result.csv'
        df_new = pd.DataFrame({'password': [input_text], 'result': "실패"})
        if os.path.exists(csv_file):
            df_existing = pd.read_csv(csv_file, converters={'features': eval})
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(csv_file, index=False)

def speak(text):
    base_voice = 'voice.mp3'
    tts = gTTS(text=text, lang='ko')
    tts.save(base_voice)
    playsound(base_voice)
    if os.path.exists(base_voice):  # 기존에 있던 파일 삭제
        os.remove(base_voice)

def process_audio_files_in_folder(folder_path, file_address, folder_name):
    wav_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')]
    recognizer = sr.Recognizer()

    for wav_file_path in wav_files:
        with sr.AudioFile(wav_file_path) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio, language='ko')
            filename = os.path.basename(wav_file_path)
            answer(text, wav_file_path, '김동현', file_address, folder_name)

            if AI_clear == 5 or AI_clear == 3:
                break

        except sr.UnknownValueError:
            print('인식 실패')  # 음성 인식 실패
        except sr.RequestError as e:
            print(f'요청 실패 : {e}')  # API KEY 오류, 네트워크 단절

def plot_similarity_graph():
    plt.figure(figsize=(10, 6))

    # 색상을 순차적으로 선택하기 위해 cycler 생성
    color_cycle = cycle(plt.cm.tab10.colors)  # tab10 팔레트 사용

    for folder_name, similarities in similarity_scores_per_folder.items():
        plt.plot(similarities, label=f'{folder_name}', color=next(color_cycle))

    plt.axhline(y=SIMILARITY_THRESHOLD, color='r', linestyle='--', label='Threshold')
    plt.title('Analyzing the Similarity of Speech')
    plt.xlabel('voice file')
    plt.ylabel('similarity')
    plt.legend()
    plt.show()

speak("복호화 시작하겠습니다.")

# 여러 폴더 경로 설정
# 이 폴더에 있는 음성들을 전부 비교하는 과정을 거침.
folder_paths = ['mincheol', 'donghyeoun', 'seungju']

# 밑에 암호화 할 파일을 정해주는 것.
file_address = 'C:\\Test\\김동현.txt'

for folder_path in folder_paths:
    process_audio_files_in_folder(folder_path, file_address, folder_path)  # 폴더 이름 전달

plot_similarity_graph()  # 모든 폴더 처리 후 그래프 생성
