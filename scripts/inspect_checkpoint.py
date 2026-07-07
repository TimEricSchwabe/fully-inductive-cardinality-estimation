import torch
import sys
from pathlib import Path

def inspect_checkpoint(checkpoint_path: str):
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"Error: File '{checkpoint_path}' does not exist.")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint safely on CPU
    try:
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    print("\n" + "="*50)
    print("           CHECKPOINT STRUCTURAL METADATA")
    print("="*50)

    if isinstance(checkpoint, dict):
        print(f"Type: PyTorch State Dictionary (dict)")
        print(f"Top-level Keys: {list(checkpoint.keys())}")
        
        # Check for typical training metadata
        for key in ['epoch', 'step', 'best_loss', 'loss', 'val_loss']:
            if key in checkpoint:
                print(f"  {key}: {checkpoint[key]}")

        # Inspect model parameters if present
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("\nModel State Dict Parameters (Weights):")
            print(f"  Total Parameter Tensors: {len(state_dict)}")
            print("\nFirst 10 Parameter Tensors (Shapes & Names):")
            for i, (name, tensor) in enumerate(state_dict.items()):
                if i >= 10:
                    print("  ...")
                    break
                print(f"  - {name:<40} Shape: {list(tensor.shape)}")
        else:
            print("\nNo 'model_state_dict' key found. Printing dictionary keys and shapes:")
            for k, v in checkpoint.items():
                if isinstance(v, torch.Tensor):
                    print(f"  - {k:<40} Shape: {list(v.shape)}")
    else:
        print(f"Type: Serialized Object (Non-dict)")
        print(f"Class Type: {type(checkpoint)}")
        if isinstance(checkpoint, torch.nn.Module):
            print("\nModule architecture:")
            print(checkpoint)

    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_checkpoint.py <path_to_model.pt>")
        sys.exit(1)
    
    inspect_checkpoint(sys.argv[1])
