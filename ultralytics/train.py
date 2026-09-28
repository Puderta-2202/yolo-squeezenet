from ultralytics import YOLO
import yaml
import os
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch

plt.rcParams["figure.figsize"] = (10, 6)

try:
    # =============================
    # 1. KONFIGURASI DATASET
    # =============================
    print("\n" + "="*70)
    print("DATASET CONFIGURATION")
    print("="*70)
    
    # Sesuaikan path dengan lokasi dataset Anda
    DATASET_BASE_PATH = "D:\\Python\\Yolo\\Data_Sekunder\\BRACOL-ORIGINAL-ANNOTATIONS\\BRACOL-ORIGINAL-DETECT\\"
    DATASET_YAML = "D:\\Python\\yolov12\\data.yaml"
    
    # Baca dan update data.yaml agar path sesuai
    if os.path.exists(DATASET_YAML):
        with open(DATASET_YAML, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update path untuk train, val, test
        data_config['path'] = DATASET_BASE_PATH
        data_config['train'] = f"{DATASET_BASE_PATH}train/images"
        data_config['val'] = f"{DATASET_BASE_PATH}valid/images"
        data_config['test'] = f"{DATASET_BASE_PATH}test/images"
        
        # Simpan ulang
        with open(DATASET_YAML, 'w') as f:
            yaml.safe_dump(data_config, f)
        
        print(f"✓ Dataset config updated: {DATASET_YAML}")
        print(f"  - Path      : {data_config.get('path')}")
        print(f"  - Train     : {data_config.get('train')}")
        print(f"  - Val       : {data_config.get('val')}")
        print(f"  - Test      : {data_config.get('test')}")
        print(f"  - Classes   : {data_config.get('nc')} ({data_config.get('names')})")
    else:
        print(f"⚠️ Warning: data.yaml tidak ditemukan di {DATASET_YAML}")
        DATASET_YAML = DATASET_BASE_PATH + "data.yaml"
        print(f"  Menggunakan: {DATASET_YAML}")
    
    # =============================
    # 2. SETTING MODEL & HYPERPARAMETER
    # =============================
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    
    MODEL = "ultralytics/cfg/models/12/yolo12n-squeezenet.yaml"
    EPOCHS = 2
    BATCH_SIZE = 32
    IMAGE_SIZE = 640
    LEARNING_RATE = 0.001
    OPTIMIZER = "Adam"
    
    print(f"Model architecture : {MODEL}")
    print(f"Epochs            : {EPOCHS}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Image size        : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Learning rate     : {LEARNING_RATE}")
    print(f"Optimizer         : {OPTIMIZER}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device            : {device}")
    
    # =============================
    # 3. LOAD & TRAINING MODEL
    # =============================
    print("\n" + "="*70)
    print("🚀 MULAI TRAINING")
    print("="*70)
    
    model = YOLO(MODEL)
    model_name = Path(MODEL).stem  # "yolo12n-squeezenet"
    
    # Hitung jumlah parameter
    print("\n🧠 INFORMASI MODEL")
    print("-"*70)
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"Total parameters      : {total_params:,}")
    print(f"Trainable parameters  : {trainable_params:,}")
    print(f"Model size            : {total_params / 1e6:.2f}M parameters")
    print("-"*70)
    
    # Jalankan training
    start_time = time.time()
    results_train = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        optimizer=OPTIMIZER,
        lr0=LEARNING_RATE,
        project="runs_yolo12_train",
        name=model_name,
        exist_ok=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=True,
        patience=50,
        save=True
    )
    train_time = time.time() - start_time
    
    run_dir = Path(results_train.save_dir)
    best_model_path = run_dir / "weights/best.pt"
    
    print(f"\n✅ Training selesai!")
    print(f"   Folder run : {run_dir}")
    print(f"   Best model : {best_model_path}")
    print(f"   Waktu     : {round(train_time, 2)} detik ({round(train_time/60, 2)} menit)")
    
    # =============================
    # 4. EVALUASI MODEL
    # =============================
    print("\n" + "="*70)
    print("🔍 EVALUASI MODEL")
    print("="*70)
    
    model_best = YOLO(best_model_path)
    val_results = model_best.val(
        data=DATASET_YAML,
        split="val",
        plots=True
    )
    
    # Ambil metrik utama
    P = val_results.results_dict.get('metrics/precision(B)', None)
    R = val_results.results_dict.get('metrics/recall(B)', None)
    m50 = val_results.results_dict.get('metrics/mAP50(B)', None)
    m5095 = val_results.results_dict.get('metrics/mAP50-95(B)', None)
    F1 = 2 * P * R / (P + R) if (P is not None and R is not None and (P+R) > 0) else None
    
    print("\n📈 METRIK EVALUASI YOLO12:")
    print("-"*70)
    print(f"Precision (P) : {P:.4f if P is not None else 'N/A'}")
    print(f"Recall (R)    : {R:.4f if R is not None else 'N/A'}")
    print(f"F1-Score      : {F1:.4f if F1 is not None else 'N/A'}")
    print(f"mAP50         : {m50:.4f if m50 is not None else 'N/A'}")
    print(f"mAP50-95      : {m5095:.4f if m5095 is not None else 'N/A'}")
    print("-"*70)
    
    # =============================
    # 5. PLOT & UPDATE results.csv
    # =============================
    print("\n" + "="*70)
    print("📊 PLOTTING & UPDATE RESULTS")
    print("="*70)
    
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"✓ results.csv ditemukan: {csv_path}")
        
        # Tambahkan kolom parameter count (jika belum ada)
        if "total_params" not in df.columns:
            df["total_params"] = total_params
        if "trainable_params" not in df.columns:
            df["trainable_params"] = trainable_params
        
        # Simpan ulang CSV
        df.to_csv(csv_path, index=False)
        print("✓ Jumlah parameter berhasil ditambahkan ke results.csv")
        
        # Plot 1: Training Loss
        try:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss", marker='o')
            if "train/cls_loss" in df.columns:
                plt.plot(df["epoch"], df["train/cls_loss"], label="Cls Loss", marker='s')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss - YOLOv12")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: mAP50-95
            plt.subplot(1, 2, 2)
            if "metrics/mAP50-95(B)" in df.columns:
                plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95", color='green', marker='o')
                plt.xlabel("Epoch")
                plt.ylabel("mAP50-95")
                plt.title("mAP50-95 per Epoch - YOLOv12")
                plt.grid(True, alpha=0.3)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(run_dir / "training_summary.png", dpi=100, bbox_inches='tight')
            print("✓ Plot disimpan: training_summary.png")
            plt.show()
        except Exception as e:
            print(f"⚠️ Warning saat plotting: {e}")
    else:
        print(f"⚠️ results.csv tidak ditemukan di {csv_path}")
    
    # =============================
    # SUMMARY
    # =============================
    print("\n" + "="*70)
    print("✅ TRAINING & EVALUASI SELESAI!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Model           : {model_name}")
    print(f"  Total params    : {total_params:,}")
    print(f"  Training time   : {round(train_time/60, 2)} menit")
    print(f"  Best model      : {best_model_path}")
    print(f"  Results folder  : {run_dir}")
    print(f"\nMetrics:")
    print(f"  Precision       : {P:.4f if P is not None else 'N/A'}")
    print(f"  Recall          : {R:.4f if R is not None else 'N/A'}")
    print(f"  mAP50-95        : {m5095:.4f if m5095 is not None else 'N/A'}")
    print("="*70)

except FileNotFoundError as e:
    print(f"\n❌ Error: File not found - {e}")
    print("Please check the model YAML path and dataset YAML path")
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
