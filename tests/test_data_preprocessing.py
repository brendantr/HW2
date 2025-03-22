import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import load_and_combine_data, preprocess_data, split_data

# Only load a sample of data for testing
def test_data_loading():
    data_dir = "eel4810-dataset"
    # Just list directories without loading all files
    print("Testing directory scanning...")
    import os
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(subdirs)} subdirectories: {subdirs}")
    
    # Load just one file from one directory as a sample
    sample_dir = os.path.join(data_dir, subdirs[0])
    import glob
    csv_files = glob.glob(os.path.join(sample_dir, "*.csv"))
    
    if csv_files:
        import pandas as pd
        print(f"Loading one sample file: {csv_files[0]}")
        df = pd.read_csv(csv_files[0], header=None)
        df.columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5']
        print(f"Sample file shape: {df.shape}")
        print(f"Sample data:\n{df.head()}")
        return True
    return False

def test_preprocessing():
    # Create a small synthetic DataFrame for testing
    import pandas as pd
    import numpy as np
    
    # Create sample data with all label types (0, 1, 2, #)
    data = {
        'Col2': [123, 156, 174, 180], 
        'Col3': [45, 52, 60, 65],
        'Col4': [90, 93, 95, 98],
        'Col5': ['0', '1', '2', '#']
    }
    df = pd.DataFrame(data)
    
    print("\nTesting preprocessing with synthetic data...")
    X, y, scaler = preprocess_data(df)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels: {y}")
    print(f"Normalized features:\n{X}")
    
    # Verify label mapping
    expected_y = np.array([0, 1, 2, 3])
    assert np.array_equal(y, expected_y), f"Label mapping failed: {y} != {expected_y}"
    print("✓ Label mapping verified")
    
    return True

def test_data_splitting():
    # Create synthetic data for splitting
    import numpy as np
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 4, size=100)
    
    print("\nTesting data splitting...")
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.1)
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    print(f"Split ratio: {len(X_val)/len(X)} (expected: 0.1)")
    
    assert len(X_val) == 10, f"Expected 10 validation samples, got {len(X_val)}"
    print("✓ Split sizes verified")
    
    return True

if __name__ == "__main__":
    test_data_loading()
    test_preprocessing()
    test_data_splitting()
    print("\nAll data preprocessing tests passed!")