import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.config import Config
from src.detection import CustomDataset, get_model

def train_model():
    # 設定をロード
    config = Config()

    # デバイスを設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットを準備
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])
    
    train_dataset = CustomDataset(config.train_data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # モデルを初期化
    model = get_model(config.model_name).to(device)

    # 損失関数とオプティマイザを定義
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学習ループ
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # パラメータの勾配をゼロにする
            optimizer.zero_grad()

            # フォワードパス
            outputs = model(images)
            loss = criterion(outputs, labels)

            # バックワードパスと最適化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # エポックごとの損失を表示
        print(f'エポック [{epoch+1}/{config.num_epochs}], 損失: {running_loss/len(train_loader):.4f}')

    # 学習済みモデルを保存
    model_save_path = os.path.join(config.model_save_path, 'trained_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'モデルを {model_save_path} に保存しました')

if __name__ == "__main__":
    train_model()