from ultralytics import YOLO
import torch

try:
    # Load model
    print("=" * 60)
    print("YOLO12 Training Configuration")
    print("=" * 60)
    
    model = YOLO("ultralytics/cfg/models/12/yolo12n-squeezenet.yaml")
    
    # Display model parameters
    print("\nModel Information:")
    print("-" * 60)
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params / 1e6:.2f}M parameters")
    print("-" * 60)
    
    # Training configuration
    print("\nTraining Configuration:")
    print("-" * 60)
    print(f"Optimizer: Adam")
    print(f"Learning Rate: 0.001")
    print(f"Epochs: 2")
    print(f"Batch Size: 32")
    print(f"Image Size: 640x640")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("-" * 60)
    
    # Train the model
    print("\nStarting Training...")
    print("-" * 60)
    results = model.train(
        data="D:\\Python\\Yolo\\Data_Sekunder\\BRACOL-ORIGINAL-ANNOTATIONS\\BRACOL-ORIGINAL-DETECT\\data.yaml",
        epochs=2,
        imgsz=640,
        batch=32,
        optimizer='Adam',
        lr0=0.001,
        patience=50,
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=True,
        save=True
    )
    
    # Display evaluation metrics
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    
    # Get validation results
    print("\nRunning Validation...")
    val_results = model.val()
    
    print("\nDetailed Results:")
    print("-" * 60)
    
    # Extract metrics safely
    if hasattr(val_results, 'results_dict') and val_results.results_dict:
        results_dict = val_results.results_dict
        
        # Try different key formats
        precision = results_dict.get('metrics/precision(B)', results_dict.get('precision', 'N/A'))
        recall = results_dict.get('metrics/recall(B)', results_dict.get('recall', 'N/A'))
        map50 = results_dict.get('metrics/mAP50(B)', results_dict.get('mAP50', 'N/A'))
        map5095 = results_dict.get('metrics/mAP50-95(B)', results_dict.get('mAP50-95', 'N/A'))
        
        if isinstance(precision, (int, float)):
            print(f"Box Precision (P): {precision:.4f}")
        else:
            print(f"Box Precision (P): {precision}")
            
        if isinstance(recall, (int, float)):
            print(f"Box Recall (R): {recall:.4f}")
        else:
            print(f"Box Recall (R): {recall}")
            
        if isinstance(map50, (int, float)):
            print(f"Box mAP50: {map50:.4f}")
        else:
            print(f"Box mAP50: {map50}")
            
        if isinstance(map5095, (int, float)):
            print(f"Box mAP50-95: {map5095:.4f}")
        else:
            print(f"Box mAP50-95: {map5095}")
    else:
        print("Validation results not available")
        
    print("-" * 60)
    
    print("\n✓ Training completed successfully!")
    print("=" * 60)
    
except FileNotFoundError as e:
    print(f"❌ Error: File not found - {e}")
    print("Please check the model YAML path and dataset YAML path")
except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
