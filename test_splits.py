#!/usr/bin/env python3
"""
Test script to verify dataset splitting functionality.
"""
from utils import get_config, create_dataset_splits, load_dataset_by_split

def test_dataset_splits():
    """Test the dataset splitting functionality"""
    print("Testing dataset splitting functionality...")
    
    # Get configuration
    config = get_config()
    
    # Create splits (this will create dataset_splits.json if it doesn't exist)
    print("\n1. Creating dataset splits...")
    train_indices, val_indices, test_indices = create_dataset_splits(config)
    
    # Load each split
    print("\n2. Loading individual splits...")
    train_data = load_dataset_by_split(config, 'train')
    val_data = load_dataset_by_split(config, 'val')
    test_data = load_dataset_by_split(config, 'test')
    
    # Verify splits
    print("\n3. Verifying splits...")
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Test examples: {len(test_data)}")
    print(f"Total: {len(train_data) + len(val_data) + len(test_data)}")
    
    # Verify no overlap
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)
    
    print(f"\n4. Checking for data leakage...")
    print(f"Train-Validation overlap: {len(train_val_overlap)} (should be 0)")
    print(f"Train-Test overlap: {len(train_test_overlap)} (should be 0)")
    print(f"Validation-Test overlap: {len(val_test_overlap)} (should be 0)")
    
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("✅ SUCCESS: No data leakage detected!")
    else:
        print("❌ ERROR: Data leakage detected!")
    
    # Show sample data
    print(f"\n5. Sample data from each split:")
    print(f"Train sample: {train_data[0]['translation'][config['lang_src']][:50]}...")
    print(f"Val sample: {val_data[0]['translation'][config['lang_src']][:50]}...")
    print(f"Test sample: {test_data[0]['translation'][config['lang_src']][:50]}...")
    
    print("\n6. Testing consistency...")
    # Load splits again to verify consistency
    train_indices_2, val_indices_2, test_indices_2 = create_dataset_splits(config)
    
    if (train_indices == train_indices_2 and 
        val_indices == val_indices_2 and 
        test_indices == test_indices_2):
        print("✅ SUCCESS: Dataset splits are consistent across calls!")
    else:
        print("❌ ERROR: Dataset splits are not consistent!")

if __name__ == "__main__":
    test_dataset_splits()
