from deepface import DeepFace
import pandas as pd
import os
import cv2

data= {
    'Age': [],
    'Gender': [],
    'Race': [],
    'File Name': [],
    'Emotion': [],
}

for file in os.listdir('images'):
    result = DeepFace.analyze(cv2.imread(f'images/{file}'), actions=('gender', 'age', 'race', 'emotion'))

    data['File Name'].append(file.split('.')[0])
    data['Age'].append(result[0]['age'])
    data['Race'].append(result[0]['dominant_race'])
    data['Gender'].append(result[0]['dominant_gender'])
    data['Emotion'].append(result[0]['dominant_emotion'])

df = pd.DataFrame(data)
print(df)
df.to_csv('people.csv')







