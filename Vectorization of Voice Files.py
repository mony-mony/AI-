from gtts import gTTS
from playsound import playsound
import time, os
import csv
from sentence_transformers import SentenceTransformer
import librosa
import numpy as np
import pandas as pd
import speech_recognition as sr

AI_lock = 1

model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def process_audio_file(audio_path):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language='ko')
        answer(text, audio_path)
    except sr.UnknownValueError:
        print('인식 실패') # 음성 인식 실패
    except sr.RequestError as e:
        print('요청 실패 : {0}'.format(e)) # API KEY 오류, 네트워크 단절

def answer(input_text, audio_path):
    # 오디오 데이터 입력
    def load_audio(file_path):
        y, sr = librosa.load(file_path, sr=None)
        return y, sr

    def extract_features(y, sr, n_mfcc=192):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean

    # 입력 음성 파일 목록
    audio_files = [audio_path]

    # 특징 추출 결과 저장 리스트
    features_list = []

    for file in audio_files:
        y, sr = load_audio(file)
        features = extract_features(y, sr, n_mfcc=192)
        features_list.append(features.tolist())

    # 특징 데이터를 DataFrame으로 변환
    df_new = pd.DataFrame({'filename': ['김동현'], 'features': features_list, 'file_path': audio_files})

    # CSV 파일 경로 설정
    csv_file = 'audio_password.csv'

    # 기존 CSV 파일이 존재하면 이어서 작성, 그렇지 않으면 새로 생성
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file, converters={'features': eval})
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # CSV 파일로 저장
    df_combined.to_csv(csv_file, index=False)

    # 텍스트 데이터 입력
    file = input_text
    file = file.replace("로", "").replace("으로", "").replace("해", "").replace("줘", "").replace(" ", "")
    
    speak(file + "로 설정하겠습니다.")

    with open('text_password.csv', 'a', newline='', encoding='cp949') as f:
        wr = csv.writer(f)
        wr.writerow(['김동현', file, file, list(model.encode(file))])

    # 파일 암호화
    c_Encode('C:\\Test\\김동현.txt', file)

    speak('완료했습니다.')

    global AI_lock 
    AI_lock = 5 

def c_Encode(address, s):
    byte_s = s.encode('utf-8')
    with open(address, 'r') as original:
        with open(address + ".cry", 'w+') as encode_file:
            n = 0
            for line in original:
                temp_line = bytearray(line, 'utf-8')
                for i in range(len(temp_line)):
                    temp_line[i] ^= byte_s[n]
                    n += 1
                    if n >= len(byte_s):
                        n = 0
                encode_file.write(temp_line.decode('utf-8'))

def speak(text):
    base_voice = 'voice.mp3'
    tts = gTTS(text=text, lang='ko')
    tts.save(base_voice)
    playsound(base_voice)
    if os.path.exists(base_voice): # 기존에 있던 파일 삭제
        os.remove(base_voice)

speak("시작하겠습니다.")

# WAV 파일 경로 설정 (예: 'input.wav')

# wav_file_path = '김동현1\안녕하세요.wav'
# process_audio_file(wav_file_path)

folder_path = 'seungju'
wav_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')]

for wav_file_path in wav_files:
    process_audio_file(wav_file_path)

while True:
    if AI_lock == 5:
        break
    else:
        time.sleep(0.05)
