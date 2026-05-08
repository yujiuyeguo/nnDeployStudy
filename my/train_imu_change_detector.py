import csv
import math
import random
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "imu_change_train.csv"
TEST_CSV = BASE_DIR / "imu_change_test.csv"
ONNX_PATH = BASE_DIR / "imu_change_detector.onnx"
PT_PATH = BASE_DIR / "imu_change_detector.pt"

CSV_COLUMNS = [
    "timestamp_ms",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "label",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_sample_csv(path: Path, sample_count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    dt = 0.02
    rows = []

    for index in range(sample_count):
        timestamp_ms = int(index * dt * 1000)
        violent = 1 if rng.random() < 0.35 else 0

        if violent:
            angle_scale = rng.uniform(28.0, 85.0)
            gyro_scale = rng.uniform(120.0, 280.0)
            wobble = rng.uniform(0.8, 1.4)
        else:
            angle_scale = rng.uniform(1.0, 16.0)
            gyro_scale = rng.uniform(5.0, 45.0)
            wobble = rng.uniform(0.2, 0.8)

        phase = index * 0.05 + rng.uniform(-0.2, 0.2)
        roll = angle_scale * math.sin(phase) + rng.normal(0.0, 1.2 + violent * 2.5)
        pitch = angle_scale * 0.8 * math.cos(phase * 1.15) + rng.normal(0.0, 1.0 + violent * 2.0)
        yaw = angle_scale * 0.6 * math.sin(phase * 0.65 + 0.5) + rng.normal(0.0, 0.8 + violent * 1.8)

        gyro_x = gyro_scale * math.sin(phase * 1.4) * wobble + rng.normal(0.0, 4.0 + violent * 12.0)
        gyro_y = gyro_scale * math.cos(phase * 1.1) * wobble + rng.normal(0.0, 4.0 + violent * 10.0)
        gyro_z = gyro_scale * math.sin(phase * 0.9 + 1.2) * wobble + rng.normal(0.0, 4.0 + violent * 14.0)

        if violent and rng.random() < 0.45:
            spike_axis = rng.integers(0, 6)
            spike_value = rng.choice([-1.0, 1.0]) * rng.uniform(30.0, 120.0)
            values = [roll, pitch, yaw, gyro_x, gyro_y, gyro_z]
            values[spike_axis] += spike_value
            roll, pitch, yaw, gyro_x, gyro_y, gyro_z = values

        rows.append(
            [
                timestamp_ms,
                round(roll, 4),
                round(pitch, 4),
                round(yaw, 4),
                round(gyro_x, 4),
                round(gyro_y, 4),
                round(gyro_z, 4),
                violent,
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            features.append(
                [
                    float(row["roll_deg"]),
                    float(row["pitch_deg"]),
                    float(row["yaw_deg"]),
                    float(row["gyro_x_dps"]),
                    float(row["gyro_y_dps"]),
                    float(row["gyro_z_dps"]),
                ]
            )
            labels.append(float(row["label"]))

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


class ImuChangeDetector(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.network(x)


def build_dataloader(features: np.ndarray, labels: np.ndarray, batch_size: int = 32) -> DataLoader:
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def evaluate(model: nn.Module, features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(features, dtype=torch.float32)
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).float().cpu().numpy()

    accuracy = float((preds == labels).mean())
    positive_recall = float((((preds == 1) & (labels == 1)).sum()) / max((labels == 1).sum(), 1))
    return accuracy, positive_recall


def train_model(train_features: np.ndarray, train_labels: np.ndarray) -> ImuChangeDetector:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0) + 1e-6

    model = ImuChangeDetector(mean, std)
    dataloader = build_dataloader(train_features, train_labels, batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, 81):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        if epoch in {1, 20, 40, 60, 80}:
            avg_loss = epoch_loss / len(dataloader.dataset)
            print(f"epoch={epoch:02d} loss={avg_loss:.4f}")

    return model


def export_onnx(model: nn.Module) -> None:
    model.eval()
    dummy_input = torch.randn(1, 6, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH.as_posix(),
        dynamo=False,
        input_names=["imu_input"],
        output_names=["change_logit"],
        dynamic_axes={"imu_input": {0: "batch_size"}, "change_logit": {0: "batch_size"}},
        opset_version=13,
    )
    onnx_model = onnx.load(ONNX_PATH.as_posix())
    onnx.checker.check_model(onnx_model)


def main() -> None:
    set_seed()
    create_sample_csv(TRAIN_CSV, sample_count=320, seed=42)
    create_sample_csv(TEST_CSV, sample_count=120, seed=99)

    train_features, train_labels = load_dataset(TRAIN_CSV)
    test_features, test_labels = load_dataset(TEST_CSV)

    model = train_model(train_features, train_labels)

    train_acc, train_recall = evaluate(model, train_features, train_labels)
    test_acc, test_recall = evaluate(model, test_features, test_labels)

    torch.save(model.state_dict(), PT_PATH)
    export_onnx(model)

    print(f"train_csv={TRAIN_CSV}")
    print(f"test_csv={TEST_CSV}")
    print(f"torch_model={PT_PATH}")
    print(f"onnx_model={ONNX_PATH}")
    print(f"train_accuracy={train_acc:.4f}, train_positive_recall={train_recall:.4f}")
    print(f"test_accuracy={test_acc:.4f}, test_positive_recall={test_recall:.4f}")
    print("输入特征顺序: roll_deg, pitch_deg, yaw_deg, gyro_x_dps, gyro_y_dps, gyro_z_dps")


if __name__ == "__main__":
    main()
