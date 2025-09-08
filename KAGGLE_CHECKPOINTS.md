# Kaggle Checkpoint System

## 🎯 **System Overview**

The checkpoint system is now optimized for Kaggle workflows:

- **📥 Load checkpoints** from Kaggle input directory (uploaded pre-trained models)
- **💾 Save checkpoints** to current working directory (new training progress)

## 📁 **Directory Structure**

### **Kaggle Input Directory:**
```
/kaggle/input/translation-checkpoints/
├── tmodel_04.pt    # Pre-trained checkpoint (epoch 4)
├── tmodel_06.pt    # Pre-trained checkpoint (epoch 6)  
└── tokenizer_*.json # Tokenizers (optional)
```

### **Current Working Directory:**
```
/kaggle/working/     # Current directory
├── weights/         # New checkpoints saved here
│   ├── tmodel_07.pt # New checkpoint (epoch 7)
│   ├── tmodel_08.pt # New checkpoint (epoch 8)
│   └── ...
├── train.py
├── test.py
├── utils.py
└── dataset_splits.json
```

## ⚙️ **Configuration**

### **In `utils.py` config:**
```python
"preload": "06",  # Load from epoch 6
"preload_input_dir": "/kaggle/input/translation-checkpoints",  # Kaggle input
```

## 🔄 **How It Works**

### **1. Loading Checkpoints (Preload)**

**Function:** `get_preload_weights_path(config, epoch)`

**Search Order:**
1. **Kaggle Input:** `/kaggle/input/translation-checkpoints/tmodel_06.pt`
2. **Local Fallback:** `./weights/tmodel_06.pt`
3. **Graceful Failure:** Returns expected path with clear error message

### **2. Saving Checkpoints (New Training)**

**Function:** `get_weights_path(config, epoch)`

**Always saves to:** `./weights/tmodel_XX.pt`

## 🚀 **Usage Examples**

### **Training (Resume from epoch 6):**
```bash
python train.py
# Will load: /kaggle/input/translation-checkpoints/tmodel_06.pt
# Will save: ./weights/tmodel_07.pt, ./weights/tmodel_08.pt, etc.
```

### **Testing:**
```bash
python test.py 6 "Tämä on testi"
# Will load: /kaggle/input/translation-checkpoints/tmodel_06.pt

python test.py 7 "Tämä on testi"  
# Will load: ./weights/tmodel_07.pt (if training continued)
```

### **Evaluation:**
```bash
python test.py 6 --evaluate
# Will evaluate using checkpoint from Kaggle input
```

## 📋 **Kaggle Setup Checklist**

### **1. Upload Pre-trained Models**
- Create a Kaggle dataset with your checkpoints
- Upload `tmodel_04.pt`, `tmodel_06.pt`, etc.
- Name the dataset: `translation-checkpoints`

### **2. Update Config (if needed)**
- Modify `preload_input_dir` if using different dataset name
- Set `preload` to the epoch you want to resume from

### **3. File Structure in Kaggle**
```
📁 Input Data:
   └── translation-checkpoints/
       ├── tmodel_06.pt
       └── tokenizer_*.json (optional)

📁 Working Directory:
   ├── train.py
   ├── test.py
   ├── utils.py
   ├── data/
   └── weights/ (created automatically)
```

## 🔍 **Error Handling**

If checkpoint not found, the system will:
1. **List available checkpoints** in both directories
2. **Show clear paths** being searched
3. **Provide helpful suggestions** for resolution

Example error output:
```
Model checkpoint not found: /kaggle/input/translation-checkpoints/tmodel_07.pt
Available checkpoints:
  Kaggle input directory:
    - tmodel_04.pt
    - tmodel_06.pt
  Local weights directory:
    - No checkpoints found
```

## ⚡ **Benefits**

✅ **Seamless Resume**: Load any pre-trained checkpoint from Kaggle input
✅ **Progress Tracking**: Save new checkpoints locally as training continues  
✅ **Version Control**: Clear separation between uploaded and generated models
✅ **Flexibility**: Can load from either location with automatic fallback
✅ **Competition Ready**: Perfect for Kaggle's input/output file system

## 🔧 **Advanced Configuration**

### **Change Input Directory:**
```python
"preload_input_dir": "/kaggle/input/my-custom-checkpoints",
```

### **Disable Preloading:**
```python
"preload": None,  # Start from scratch
```

### **Force Local Loading:**
```python
"preload_input_dir": "./weights",  # Load from local instead
```

Your checkpoint system is now Kaggle-optimized! 🎉
