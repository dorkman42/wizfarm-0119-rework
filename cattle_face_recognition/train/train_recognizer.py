"""
소 얼굴 인식 모델 학습 스크립트
ResNet + ArcFace Loss 기반 Metric Learning

상업적 사용 가능 (PyTorch BSD, torchvision BSD)
"""
import os
import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm
import math


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss (Additive Angular Margin Loss)
    상업적 사용 가능한 구현
    """
    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight)
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Threshold
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class CattleFaceEmbedder(nn.Module):
    """
    소 얼굴 특징 추출 모델
    ResNet 백본 + Embedding Layer
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        embedding_size: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()

        # 백본 선택
        if backbone == "resnet18":
            base = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            in_features = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            in_features = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            in_features = 2048
        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            in_features = 1280
        else:
            raise ValueError(f"지원하지 않는 백본: {backbone}")

        # 마지막 FC 레이어 제거
        if "resnet" in backbone:
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        else:
            self.backbone = nn.Sequential(*list(base.children())[:-1])

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        self.embedding_size = embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return embeddings

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """추론용: 정규화된 임베딩 반환"""
        embeddings = self.forward(x)
        return F.normalize(embeddings, p=2, dim=1)


class CattleFaceDataset(Dataset):
    """
    소 얼굴 데이터셋
    폴더 구조: root/cattle_id/image.jpg
    """
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or self._default_transform()

        # 이미지 목록 구성
        self.samples = []
        self.class_to_idx = {}

        for idx, cattle_dir in enumerate(sorted(self.root_dir.iterdir())):
            if not cattle_dir.is_dir():
                continue

            cattle_id = cattle_dir.name
            self.class_to_idx[cattle_id] = idx

            for img_path in cattle_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((str(img_path), idx))

        self.num_classes = len(self.class_to_idx)
        print(f"데이터셋 로드: {len(self.samples)}개 이미지, {self.num_classes}개 클래스")

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform():
    """학습용 데이터 증강"""
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomCrop((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    """검증용 변환"""
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_epoch(
    model: nn.Module,
    arcface: ArcFaceLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """한 에폭 학습"""
    model.train()
    arcface.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        embeddings = model(images)
        logits = arcface(embeddings, labels)
        loss = F.cross_entropy(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    arcface: ArcFaceLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """검증"""
    model.eval()
    arcface.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            logits = arcface(embeddings, labels)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return total_loss / len(dataloader), accuracy


def train_recognizer(
    train_dir: str,
    val_dir: Optional[str] = None,
    backbone: str = "resnet50",
    embedding_size: int = 512,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = "auto",
    output_dir: str = "runs/recognition",
    name: str = "cattle_recognizer",
):
    """
    소 얼굴 인식 모델 학습

    Args:
        train_dir: 학습 데이터 디렉토리 (cattle_id별 폴더)
        val_dir: 검증 데이터 디렉토리
        backbone: 백본 모델
        embedding_size: 임베딩 차원
        epochs: 학습 에폭
        batch_size: 배치 크기
        lr: 학습률
        device: 디바이스
        output_dir: 출력 디렉토리
        name: 실험 이름
    """
    print("=" * 50)
    print("소 얼굴 인식 모델 학습")
    print("=" * 50)

    # 디바이스 설정
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    print(f"디바이스: {device}")

    # 출력 디렉토리
    save_dir = Path(output_dir) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 데이터셋
    train_dataset = CattleFaceDataset(train_dir, transform=get_train_transform())
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = None
    if val_dir:
        val_dataset = CattleFaceDataset(val_dir, transform=get_val_transform())
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

    # 모델
    model = CattleFaceEmbedder(
        backbone=backbone,
        embedding_size=embedding_size,
        pretrained=True,
    ).to(device)

    arcface = ArcFaceLoss(
        embedding_size=embedding_size,
        num_classes=train_dataset.num_classes,
    ).to(device)

    # 옵티마이저
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=lr,
        weight_decay=0.01,
    )

    # 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 학습
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        train_loss = train_epoch(model, arcface, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        if val_loader:
            val_loss, val_acc = validate(model, arcface, val_loader, device)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Best model 저장
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "embedding_size": embedding_size,
                    "backbone": backbone,
                    "num_classes": train_dataset.num_classes,
                    "class_to_idx": train_dataset.class_to_idx,
                }, save_dir / "best.pt")
                print(f"Best model saved! (Acc: {val_acc:.2f}%)")

        scheduler.step()

        # 주기적 저장
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "embedding_size": embedding_size,
                "backbone": backbone,
                "num_classes": train_dataset.num_classes,
                "class_to_idx": train_dataset.class_to_idx,
            }, save_dir / f"epoch_{epoch+1}.pt")

    # 최종 모델 저장
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "embedding_size": embedding_size,
        "backbone": backbone,
        "num_classes": train_dataset.num_classes,
        "class_to_idx": train_dataset.class_to_idx,
    }, save_dir / "last.pt")

    print("=" * 50)
    print("학습 완료!")
    print(f"모델 저장 위치: {save_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="소 얼굴 인식 모델 학습")
    parser.add_argument("--train-dir", type=str, required=True, help="학습 데이터 디렉토리")
    parser.add_argument("--val-dir", type=str, default=None, help="검증 데이터 디렉토리")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"])
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="runs/recognition")
    parser.add_argument("--name", type=str, default="cattle_recognizer")

    args = parser.parse_args()

    train_recognizer(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        backbone=args.backbone,
        embedding_size=args.embedding_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        name=args.name,
    )


if __name__ == "__main__":
    main()
