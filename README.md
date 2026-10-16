# Caltech-101 Image Classification with HOG + NN / KNN

Classifies images from 10 [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) categories using Histogram of Oriented Gradients (HOG) features and two simple classifiers: Nearest Neighbour (NN) and K-Nearest Neighbour (KNN, K = 5). Written in MATLAB.

This is undergraduate coursework from **CSE489 Machine Vision** at the Faculty of Engineering, Ain Shams University (August 2019). The original assignment also asked for a multi-class SVM; that part was not completed and is not included.

## Results

Evaluated on 254 test images across 10 classes, with 465 training images. Random guessing would score 10%.

| Classifier | Accuracy |
|---|---|
| NN | **67.7%** |
| KNN (K = 5) | **70.9%** |

| Class | Test images | NN | KNN |
|---|---:|---:|---:|
| sunflower | 35 | 74.3% | 88.6% |
| crab | 28 | 71.4% | 82.1% |
| chair | 17 | 47.1% | 47.1% |
| cellphone | 14 | 78.6% | 78.6% |
| dolphin | 20 | 35.0% | 30.0% |
| stop sign | 19 | 73.7% | 57.9% |
| helicopter | 38 | 60.5% | 73.7% |
| lamp | 16 | 50.0% | 37.5% |
| kangaroo | 36 | 75.0% | 77.8% |
| laptop | 31 | 90.3% | 90.3% |

KNN helps most on classes with varied appearance (sunflower, crab, helicopter), where a single closest match is often misleading. It hurts on small, distinctive classes (stop sign, lamp), where the 5-neighbour vote gets outweighed by larger classes. Dolphin is the weakest class: at 60×50 pixels, its HOG features are often closer to crab or kangaroo than to other dolphins.

## How it works

1. **Preprocessing.** Each image is resized to 60×50 pixels.
2. **Feature extraction.** MATLAB's `extractHOGFeatures` turns each image into a 1,080-value HOG descriptor, which captures the distribution of edge directions across the image.
3. **Classification.** For each test image, the Euclidean distance to every training image is computed.
   - **NN** assigns the class of the single closest training image.
   - **KNN** takes the 5 closest training images and assigns the most common class among them.

## Running it

The precomputed HOG features are in `features/`, so the classifier runs without the image dataset.

**MATLAB** (R2016b or later, no toolboxes needed):

```matlab
cd src
classify_nn_knn
```

**Python cross-check** (gives identical results without MATLAB):

```bash
pip install numpy scipy
python scripts/evaluate.py
```

**Regenerating the features (optional).** Download Caltech-101 and extract it so the image folders sit in `data/101_ObjectCategories/`. Then run `src/extract_hog_features.m` (needs the Computer Vision Toolbox). It uses a plain 70/30 split per class, so the numbers will differ slightly from the table above, which uses the 2019 split.

## Repository layout

```
src/
  classify_nn_knn.m        NN and KNN classification, accuracy, confusion matrix
  load_features.m          Robust loader for the feature files
  extract_hog_features.m   Rebuilds features/ from the Caltech-101 images
scripts/
  evaluate.py              Python cross-check of the classifier
features/                  HOG features from 2019, one train and one test file per class
```

## Fixes since the original 2019 version

The original code is preserved unchanged under the git tag [`v2019-original`](../../tree/v2019-original). That version ran, but its output was not meaningful because of these bugs, which this version fixes:

- **Training data was silently overwritten.** Several feature files stored their data under the wrong variable name, `sunflower_featurevector`. Loading them one after another into the workspace replaced the real sunflower features, so the "sunflower" training set actually used was the buddha category. The new `load_features.m` loads each file by its contents, not by variable name.
- **Sunflower was counted twice.** It appeared as both class 1 and class 6, so there were only 9 real classes. Class 6 is now laptop, whose test features already existed but were never used.
- **Test sets contained empty entries.** The extraction loop wrote test images into slots 46 or 51 onwards, leaving earlier slots filled with zeros. Those blank vectors were classified alongside the real images. They are now dropped.
- **No accuracy was computed.** The script stored predictions but never compared them with the true labels. It now reports overall and per-class accuracy and a confusion matrix.
- **Cleanup.** Features for five categories that were loaded but never used (car, seahorse, motorbikes, starfish, buddha) were removed, and the copy-pasted per-class loops were replaced with a single loop.
