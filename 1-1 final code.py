#드라이브와 연결, 확인
!pip install pydub
from pydub import AudioSegment
from google.colab import drive
drive.mount('/content/drive')
!mkdir drive/MyDrive/output_segments
!ls -alH drive/MyDrive/'John K - parachute.mp3'

#import
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# 음원 자르기 
def split_to_segments(source_file, output_folder, segment_len_seconds=10):
  sound = AudioSegment.from_mp3(source_file)
  os.makedirs(output_folder, exist_ok=True)
  segment_len_ms = segment_len_seconds * 1000
  num_segments = int(len(sound) / segment_len_ms)
  for i in range(num_segments):
    start_time = i * segment_len_ms
    end_time = start_time + segment_len_ms
    segment = sound[start_time:end_time]
    output_file = f"{output_folder}/저장할 파일 이름.wav"
    segment.export(output_file, format="wav")
source_file = "drive/MyDrive/원본 음원 이름"
output_folder = "drive/MyDrive/저장할 폴더 이름"
split_to_segments(source_file, output_folder)

#스펙트로그램 저장
def f(i):
    audio_file_path = f"drive/MyDrive/폴더 이름/파일 이름.wav"
    new_folder_path = "drive/MyDrive/저장할 폴더 이름"
    os.makedirs(new_folder_path, exist_ok=True)
    audio_data, sample_rate = librosa.load(audio_file_path)
    spectrogram = librosa.stft(audio_data)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=10), sr=sample_rate, y_axis='log', x_axis='time')
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+5.f dB")
    plt.title("Spectrogram")
    output_file_path = os.path.join(new_folder_path, f'summer_{i}_spectrogram.png')
    plt.savefig(output_file_path)
for i in range(시작값 , 종료값 + 1):
    f(i)

#white noise
def add_noise(audio_path, output_dir, noise_ratio):
    audio, sample_rate = sf.read(audio_path)
    if len(audio.shape) == 1:
        noise = np.random.rand(len(audio))
    else:
        noise = np.random.rand(len(audio), audio.shape[1])
    noise = noise_ratio * noise
    noisy_audio = audio + noise
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    os.makedirs(output_dir, exist_ok=True)
    audio_filename = os.path.basename(audio_path)
    output_path = os.path.join(output_dir, audio_filename)
    sf.write(output_path, noisy_audio, sample_rate)
audio_path = "drive/MyDrive/폴더 이름/파일 이름.wav"
output_dir = "drive/MyDrive/저장할 폴더 이름"
noise_ratio = 0.1 
add_noise(audio_path, output_dir, noise_ratio)

#hum noise
def add_random_hum_noise(audio_path, output_dir, noise_freq=60, noise_amplitude=0.1, noise_duration=1.0):
    audio, sample_rate = sf.read(audio_path)
    duration = len(audio) / sample_rate
    t = np.linspace(0, noise_duration, int(sample_rate * noise_duration), endpoint=False)
    hum_noise = noise_amplitude * np.sin(2 * np.pi * noise_freq * t)
    start_idx = np.random.randint(0, len(audio) - len(hum_noise))
    if len(audio.shape) == 2:  
        hum_noise = np.column_stack((hum_noise, hum_noise))
    noisy_audio = np.copy(audio)
    noisy_audio[start_idx:start_idx+len(hum_noise)] += hum_noise
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    os.makedirs(output_dir, exist_ok=True)
    audio_filename = os.path.basename(audio_path)
    output_path = os.path.join(output_dir, audio_filename)
    sf.write(output_path, noisy_audio, sample_rate)
for i in range(1,1001):
    audio_path = f"drive/MyDrive/폴더 이름/파일 이름.wav"
    output_dir = "drive/MyDrive/저장할 폴더 이름"
    add_random_hum_noise(audio_path, output_dir, noise_freq=60, noise_amplitude=0.1)

#인공지능 알고리즘
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm.auto import tqdm
from torchvision import io, transforms
class ImageDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        image = io.read_image(file_path)
        image = image.float() / 255.0
        image = transforms.functional.crop(image, 10, 75, 370, 397)
        image = image[:3]
        label = 0 if file_path.split('.')[-2].endswith('_0') else 1
        onehot_encoded = torch.zeros(2)
        onehot_encoded[label] = 1
        return image, onehot_encoded

batch_size = 32 
ds_train = ImageDataset('학습 데이터셋 경로')
ds_test = ImageDataset('테스트 데이터셋 경로')
loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        iden = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + iden)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, 0),
            nn.BatchNorm2d(out_channels)
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        iden = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + self.downsample(iden))
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 7, 2, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        self.blocks = nn.Sequential(
            ResBlock(16),
            ResBlock(16),
            Bottleneck(16, 32),
            ResBlock(32),
            ResBlock(32),
            Bottleneck(32, 64),
            ResBlock(64),
            ResBlock(64),
        )

        self.readout = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.apply(init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = self.blocks(x)
        x = self.readout(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
summary(model, (3, 224, 224), batch_size=batch_size)

learning_rate = 0.03
epochs = 5

CE_LOSS = nn.CrossEntropyLoss().to(device)
opt = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    miss = 0  # Changed train_loss to miss

    for x, y in tqdm(loader_train, desc=f'Training Epoch {epoch+1}'):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = CE_LOSS(pred, y)
        miss += loss.item() * x.size(0)  

        opt.zero_grad()
        loss.backward()
        opt.step()