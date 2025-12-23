# VTSC

This project is based on the **the VTSC framework** , which supports semantic feature encoding and allows flexible replacement with **custom downstream task networks**
. the paper is “ Efficient Semantic Codec for Real-time Vibrotactile Transmission” ，accepted by acmmm 2025.

---

## 1. Environment Setup

### 1.1 Python Version
```bash
Python == 3.10
```

1.2 Install Dependencies

Run the following command in the project root directory:

```bash
pip install -r requirements.txt
```
2. Project Structure

The project is organized using two subfolders as follows:
```bash
.
├── model_t/
│   └── init.py          # Model definitions (VTSC, downstream networks, loss function)
│   └── loss_t.py 
│   └── tsm_fram.py 
│   └── downstream_networks.py 
├── data
│   └── TestDataFile     
│       └──source_tapping
│   └── TrainDataFile 
│       └──source_tapping     
│   └── LMT108 
├── requirements.txt
└── source_files_paths_test.txt
└── README.md
```

3. Model Training

Run the following command in the project root directory:

```bash
python train.py
```

4. Model Evaluation

4.1 Evaluation

```bash
python scripts/test_codec.py
```

5. Replacing the Downstream Task Network (Important)

5.1 Using Your Own Downstream Network

In train.py and test_codec.py, the default import is:



Step 1: Define Your Network in models/model_t.py

Step 2: Modify the Import Statement

Replace the MFCM4 in following line in scripts/train.py and scripts/test_codec.py:
```bash
from models.model_t import MFCM4, coding_feat_epoch_entropy_2
```

Step 3: Replace the Model Instantiation

```bash
model = MFCM4(...)
```


6. Contact

Author: Runjie Wang
Email: wangrunjie2023@163.com
