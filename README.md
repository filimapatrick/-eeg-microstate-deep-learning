# Deep Learning Based Classification of Task-Modulated EEG Microstates During Face Perception

[![Dataset](https://img.shields.io/badge/Dataset-OpenNeuro%20ds002718-blue)](https://openneuro.org/datasets/ds002718)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen)](https://python.org)
[![License](https://img.shields.io/badge/License-CC0-lightgrey)](LICENSE)

## Overview

This project will investigate task-modulated EEG microstate dynamics during face perception and evaluate whether deep learning models can reliably classify microstates compared to classical clustering approaches. The research will utilize the publicly available Wakeman & Henson face processing EEG dataset (OpenNeuro ds002718) to bridge classical microstate analysis with modern deep learning methodologies.

## Research Objectives

### General Objective
To model and classify EEG microstates during a face perception task using both classical microstate analysis and deep learning–based neural network models.

### Specific Objectives
1. **Extract canonical EEG microstates** from task-based EEG data using Global Field Power (GFP)–guided k-means clustering
2. **Characterize microstate dynamics** across different task-related temporal windows (pre-stimulus, early perceptual, and late cognitive)
3. **Develop convolutional neural network (CNN) models** for automated classification of EEG microstates
4. **Compare deep learning–based microstate classification performance** with classical microstate labeling methods
5. **Evaluate the consistency, temporal properties, and transition dynamics** of microstates derived from classical and deep learning approaches

## Dataset Description

### Source
- **Dataset**: OpenNeuro ds002718 - "Face processing EEG dataset for EEGLAB"
- **Original Study**: Wakeman, D., Henson, R. A multi-subject, multi-modal human neuroimaging dataset. Sci Data 2, 150001 (2015)
- **Format**: BIDS-compliant EEG data in EEGLAB format

### Experimental Design
- **Participants**: 18 healthy adults (subjects 002-019)
- **Task**: Face recognition with symmetry judgment
- **Stimuli**: 
  - 150 famous faces (f001.bmp - f150.bmp)
  - 150 unfamiliar faces (u001.bmp - u150.bmp) 
  - 150 scrambled faces (s001.bmp - s150.bmp)
- **Paradigm**: Faces presented for 800-1000ms with immediate, early, and late repetitions
- **Total trials**: ~26,640 across all subjects (~1,480 per subject)

### Technical Specifications
- **EEG Channels**: 70 channels (extended 10-10% system)
- **Additional Channels**: 2 EOG, 4 miscellaneous
- **Sampling Rate**: 250 Hz
- **Recording Duration**: ~50 minutes per subject
- **Reference**: Nose electrode
- **Equipment**: Easycap system

## Methodology

### 1. EEG Preprocessing
The preprocessing pipeline will include:
- Band-pass filtering (1–40 Hz)
- Average re-referencing
- Artifact removal
- Epoch extraction around stimulus events

**Tools**: EEGLAB, MNE-Python

### 2. Time Window Analysis
EEG data will be segmented into three critical time windows:
- **Pre-stimulus baseline**: -200 to 0 ms
- **Early perceptual processing**: 0 to 300 ms  
- **Late cognitive processing**: 300 to 700 ms

### 3. Classical Microstate Analysis
- **GFP Computation**: Calculate Global Field Power for each epoch
- **Peak Detection**: Identify local GFP maxima
- **Clustering**: Apply k-means clustering (k=4, k=6) to scalp topographies at GFP peaks
- **Temporal Analysis**: Compute duration, coverage, and transition probabilities

### 4. Deep Learning Classification
#### CNN Architecture
The convolutional neural network will include:
- Spatial convolutional layers across EEG channels
- Batch normalization and activation functions
- Global average pooling
- Fully connected layers with softmax output

#### Training Strategy
- **Framework**: PyTorch/TensorFlow
- **Validation**: Subject-wise cross-validation
- **Input**: EEG topographies at GFP peaks
- **Output**: Microstate class predictions

### 5. Evaluation Metrics
- Classification accuracy
- Confusion matrices
- Agreement with classical microstate labels
- Microstate temporal properties comparison
- Transition dynamics analysis

## Installation

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Dependencies
```bash
# Install core packages
pip install numpy pandas matplotlib seaborn
pip install mne eeglab scipy scikit-learn
pip install torch torchvision tensorflow
pip install jupyter notebook

# Optional: for enhanced visualizations
pip install plotly nibabel
```

## Project Structure
```
ds002718-download/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                        # Original BIDS dataset
│   ├── sub-002/
│   ├── sub-003/
│   └── ...
├── code/
│   ├── preprocessing/           # EEG preprocessing scripts
│   ├── microstate_analysis/     # Classical microstate analysis
│   ├── deep_learning/          # CNN model development
│   └── evaluation/             # Comparison and evaluation
├── results/
│   ├── microstates/            # Classical analysis results
│   ├── cnn_models/             # Trained deep learning models
│   └── comparisons/            # Performance comparisons
└── notebooks/                  # Jupyter notebooks for analysis
```

## Expected Workflow

### Phase 1: Data Preparation
1. **Load BIDS-formatted dataset** and extract EEG data
2. **Import channel locations** from BIDS electrodes.tsv files
3. **Apply standardized preprocessing pipeline** across subjects
4. **Epoch EEG data** around stimulus onset by condition (famous, unfamiliar, scrambled)
5. **Segment epochs** into predefined temporal windows
6. **Compute Global Field Power (GFP)** and identify GFP peaks within each window

### Phase 2: Classical Microstate Analysis
1. **Extract EEG scalp topographies** at GFP peaks
2. **Perform k-means clustering** to derive canonical microstate maps (k = 4; exploratory k = 6)
3. **Assign microstate labels** to continuous EEG data
4. **Compute microstate temporal parameters** (mean duration, coverage, transition probabilities)
5. **Compare microstate dynamics** across stimulus conditions and temporal windows

### Phase 3: Deep Learning Model Development
1. **Construct labeled datasets** using classical microstate assignments as training targets
2. **Design a CNN architecture** to classify EEG scalp topographies into microstate classes
3. **Implement subject-wise cross-validation** to prevent data leakage
4. **Train models** using supervised learning and optimize hyperparameters via validation performance

### Phase 4: Evaluation and Comparison
1. **Evaluate CNN classification performance** using accuracy and confusion matrices
2. **Quantify agreement** between classical and CNN-derived microstate labels
3. **Compare temporal dynamics** (duration, coverage, transitions) between methods
4. **Perform statistical testing** to assess performance differences and robustness

## Expected Outcomes

### Scientific Contributions
- **Methodological advancement** in microstate analysis using deep learning
- **Task-based EEG microstate characterization** during face perception
- **Performance comparison** between classical and modern approaches
- **Open-source pipeline** for reproducible microstate analysis

### Publications and Dissemination
- Peer-reviewed manuscript in computational neuroscience journal
- Conference presentations at relevant scientific meetings
- Open-source code repository for community use
- Tutorial materials for educational purposes

## Significance

This research will contribute to multiple domains:

1. **Computational Neuroscience**: Novel application of deep learning to EEG microstate analysis
2. **Cognitive Science**: Enhanced understanding of face processing neural dynamics
3. **Methodological Development**: Bridging classical and modern analytical approaches
4. **Reproducible Research**: Open dataset usage ensuring transparency and replicability
5. **NeuroAI Applications**: Informing future AI applications in brain dynamics modeling

## Timeline

- **Months 1-2**: Data preprocessing and classical microstate analysis
- **Months 3-4**: CNN model development and training
- **Months 5-6**: Evaluation, comparison, and statistical analysis  
- **Months 7-8**: Manuscript preparation and code documentation

## Citation

When using this work, please cite:

```bibtex
@article{wakeman2015multi,
  title={A multi-subject, multi-modal human neuroimaging dataset},
  author={Wakeman, Daniel G and Henson, Richard N},
  journal={Scientific data},
  volume={2},
  number={1},
  pages={1--10},
  year={2015},
  publisher={Nature Publishing Group}
}
```

## License

This project will be released under CC0 license, consistent with the original dataset licensing.

## Contact

For questions about this research project, please open an issue in this repository or contact the research team.

---

**Note**: This research project is currently in planning phase. All methodological descriptions use future tense to indicate planned rather than completed work.# -eeg-microstate-deep-learning
