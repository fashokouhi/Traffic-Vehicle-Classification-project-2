# Traffic Vehicle Classification

## Reliable Traffic-Vehicle Classification with CNNs and Transfer Learning

این پروژه یک سیستم طبقه‌بندی تصاویر وسایل نقلیه در ترافیک است. روند پیاده‌سازی شامل ممیزی و پاک‌سازی تصاویر، ساخت CNN پایه، اجرای آزمایش‌های مقایسه‌ای، بررسی چند معماری pretrained، fine-tuning مدل ResNet18، ارزیابی نهایی، تحلیل confidence و پیاده‌سازی اسکریپت پیش‌بینی با خروجی JSON است.

## Final Results

| Metric | Result |
|---|---:|
| Training images | 1,537 |
| Validation images | 385 |
| Test images | 400 |
| Best Validation Accuracy | 97.40% |
| Test Accuracy | 96.75% |
| Test Macro Precision | 96.94% |
| Test Macro Recall | 96.51% |
| Test Macro F1 | 96.65% |
| Selected Confidence Threshold | 0.70 |

نتایج بالا بر اساس آخرین خروجی‌های ثبت‌شده در Notebook آموزش و Notebook ارزیابی پروژه هستند.

---

## 1. Classes

مدل نهایی هشت کلاس زیر را پیش‌بینی می‌کند:

```text
0 -> ambulance
1 -> autobus
2 -> kamyun
3 -> kamyunet
4 -> minibus
5 -> savari
6 -> taxi
7 -> vanet
```

در پوشه `unclean - merge`، پوشه‌ای با نام `neysan` نیز بررسی شد. این کلاس در آموزش مدل وارد نشده است؛ بنابراین مدل تصاویر آن را به یکی از هشت کلاس موجود نسبت می‌دهد و در صورت پایین بودن confidence، مقدار `needs_review` را فعال می‌کند.

---

## 2. Dataset Preparation

داده‌ها ابتدا در پوشه‌های ادغام‌شده بررسی شدند و پس از بازبینی و پاک‌سازی، مجموعه تمیزشده برای آموزش و اعتبارسنجی استفاده شد.

ساختار کلی داده‌های محلی پروژه:

```text
data/
├── train - merge/
├── train - merge - clean/
├── test/
└── unclean - merge/
    └── neysan/
```

نام دقیق پوشه‌ها ممکن است در نسخه محلی مخزن متفاوت باشد.

### Data Audit

در Notebook ممیزی داده، مراحل زیر انجام شدند:

- بررسی ساختار پوشه‌ها و پسوند فایل‌ها
- بررسی قابل‌خواندن بودن تصاویر
- بررسی ابعاد تصاویر
- محاسبه hash برای بررسی تصاویر تکراری
- بررسی تصاویر مشکوک و برچسب‌های نادرست
- بازبینی دستی نمونه‌های مشکوک
- بررسی تصاویر مشابه با perceptual hash (pHash)

خلاصه تصمیم‌های ثبت‌شده در بازبینی دستی:

| Decision | Count |
|---|---:|
| `wrong_label` | 55 |
| `keep` | 6 |
| `ambiguous` | 5 |
| `unrelated` | 5 |
| `near_duplicate` | 5 |

در بررسی exact hash ثبت‌شده در Notebook، hash تکراری پیدا نشد. برای دو تصویر مشابه نیز فاصله Hamming در pHash برابر 14 ثبت شد.

### Train / Validation Split

در Notebook پاک‌سازی، مجموعه Train و Validation با استفاده از `train_test_split` ساخته شد. نسبت Validation برابر 20 درصد، مقدار `random_state=42` و گزینه `stratify=y` استفاده شدند. مسیر تصاویر و labelها در فایل `vehicle_split.pkl` ذخیره شدند.

```text
Train:       1537
Validation:   385
Test:         400
Random seed:   42
```

مجموعه Test جداگانه برای ارزیابی مدل استفاده شد.

---

## 3. DataLoader

برای بارگذاری تصاویر، کلاس `VehicleDataset` در فایل `src/data_loader.py` پیاده‌سازی شد.

این کلاس مسیر تصاویر، labelها و transform را دریافت می‌کند و تصویر را به RGB تبدیل می‌کند. سپس در صورت وجود transform، آن را روی تصویر اعمال می‌کند.

در پیاده‌سازی DataLoader، اندازه تصاویر به `224 × 224` تغییر داده شد و با `ToTensor()` به tensor تبدیل شدند.

تنظیمات اصلی:

```text
Image size: 224 × 224
Batch size: 32
Training shuffle: True
Validation shuffle: False
Num workers: 0
```

---

## 4. CNN Baseline

برای ایجاد یک نقطه مرجع، یک CNN ساده از ابتدا ساخته و آموزش داده شد.

معماری مدل:

```text
Input image (3 × 224 × 224)
        |
Conv2d: 3 -> 32
        |
ReLU
        |
MaxPool2d
        |
Conv2d: 32 -> 64
        |
ReLU
        |
MaxPool2d
        |
Flatten
        |
Linear: 200704 -> 128
        |
ReLU
        |
Linear: 128 -> 8
```

در اجرای ثبت‌شده، از `CrossEntropyLoss` و optimizer از نوع Adam با learning rate برابر `0.001` استفاده شد.

نتیجه بهترین اجرای CNN پایه:

```text
Best CNN Baseline Validation Accuracy: 78.44%
```

این نتیجه به‌عنوان نقطه مرجع برای مقایسه آزمایش‌های CNN ساده استفاده شد.

---

## 5. Controlled Experiments

در Notebook آموزش، چند آزمایش برای بررسی تأثیر تنظیمات مختلف روی عملکرد CNN اجرا شد.

### 5.1 Dropout

مقادیر مختلف Dropout بررسی شدند:

| Dropout Probability | Best Validation Accuracy |
|---:|---:|
| 0.0 | 75.32% |
| 0.3 | 75.58% |
| 0.5 | 75.58% |

در این اجراها، افزایش Dropout به بهبود قابل‌توجهی منجر نشد.

### 5.2 Max Pooling vs Average Pooling

نتایج ثبت‌شده:

```text
Max-pooling baseline: 78.44%
Average-pooling run:  76.36%
```

در این مقایسه، اجرای Max Pooling نتیجه بهتری داشت.

### 5.3 Weight Decay

پارامتر زیر بررسی شد:

```text
weight_decay = 1e-4
```

نتیجه ثبت‌شده:

```text
Best Validation Accuracy: 76.88%
```

### 5.4 Learning Rate Scheduler

از `StepLR` با تنظیمات زیر استفاده شد:

```text
step_size = 3
gamma = 0.1
```

بهترین Validation Accuracy در این اجرا برابر بود با:

```text
Best Validation Accuracy: 72.73%
```

---

## 6. Simulated Class Imbalance and Balanced Batches

برای بررسی اثر عدم‌توازن کلاس‌ها، یک مجموعه آموزشی نامتوازن به‌صورت مصنوعی ساخته شد.

تعداد نمونه‌ها از 1,537 به 985 کاهش یافت.

توزیع نمونه‌های مجموعه نامتوازن:

| Class | Count |
|---|---:|
| ambulance | 174 |
| autobus | 199 |
| kamyun | 48 |
| kamyunet | 68 |
| minibus | 179 |
| savari | 59 |
| taxi | 59 |
| vanet | 199 |
| **Total** | **985** |

برای آزمایش Balanced Batches از `BalancedBatchSampler` با `batch_size=32` استفاده شد. در هر batch چهار نمونه از هر کلاس قرار گرفت.

### Results

| Strategy | Validation Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Standard batches | 68.05% | 69.37% | 68.52% | 66.96% |
| Balanced batches | 70.65% | 72.39% | 70.53% | 69.48% |

در این آزمایش، Balanced Batching معیارهای کلی بالاتری نسبت به Standard Batches ثبت کرد؛ بااین‌حال، تغییر Recall برای تمام کلاس‌ها یکسان نبود.

---

## 7. Cross-Entropy vs BCE

دو روش محاسبه loss بررسی شدند:

- `CrossEntropyLoss`
- `BCEWithLogitsLoss`

در آزمایش BCE، labelها به one-hot با نوع `float` تبدیل شدند. برای محاسبه کلاس پیش‌بینی‌شده نیز از `argmax(logits)` استفاده شد.

نتایج ثبت‌شده:

| Loss | Validation Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Cross-Entropy baseline | 78.44% | 78.17% | 78.13% | 78.01% |
| BCE | 79.22% | 78.94% | 78.95% | 78.79% |

در این اجرا، BCE اندکی بهتر از Cross-Entropy baseline عمل کرد. برای مدل نهایی Transfer Learning از `CrossEntropyLoss` استفاده شد.

---

## 8. Transfer Learning

در ادامه، چند معماری pretrained بررسی شدند. تنظیمات آموزش در همه اجراها یکسان نبود؛ بنابراین نتایج زیر خلاصه‌ای از بهترین Validation Accuracy ثبت‌شده در runهای موجود هستند.

| Architecture / Experiment | Best Validation Accuracy |
|---|---:|
| ResNet18 - Initial run | 89.87% |
| ResNet18 - Fine-tuning layer 4 | 95.32% |
| ResNet18 - Fine-tuning layer 3 | 96.10% |
| ResNet50 - Initial run | 89.61% |
| ResNet50 - Fine-tuning layer 4 | 91.43% |
| ResNet34 - Initial run | 87.27% |
| ResNet34 - Fine-tuning layer 4 | 92.73% |
| EfficientNet-B0 - Initial run | 88.83% |
| EfficientNet-B0 - Fine-tuning | 90.39% |
| MobileNetV3-Large - Initial run | 88.83% |
| MobileNetV3-Large - Fine-tuning | 91.43% |
| DenseNet121 - Best recorded run | 91.69% |
| ConvNeXt-Tiny - Best recorded run | 94.55% |
| ResNet18 - Learning-rate experiments | 97.14% |

پس از این آزمایش‌ها، تمرکز آموزش نهایی روی ResNet18 قرار گرفت.

---

## 9. ResNet18 Fine-Tuning

برای آموزش مدل نهایی، از ResNet18 با وزن‌های pretrained روی ImageNet استفاده شد. لایه classifier برای پیش‌بینی هشت کلاس تغییر کرد و بخش‌های `layer3`، `layer4` و `fc` قابل آموزش شدند.

تنظیمات مدل:

```text
Architecture:       ResNet18
Pretrained weights: ImageNet
Trainable layers:   layer3, layer4, fc
Loss:               CrossEntropyLoss
Optimizer:          Adam
Learning rate:      2e-5
Epochs:             15
Input size:         224 × 224
Batch size:         32
```

### Data Augmentation

چند تنظیم مختلف برای augmentation بررسی شدند.

در یکی از اجراها، transform آموزش شامل موارد زیر بود:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor()
])
```

همچنین یک اجرای جداگانه با چرخش قوی‌تر و تغییر perspective انجام شد:

```python
transforms.RandomRotation(40)

transforms.RandomPerspective(
    distortion_scale=0.3,
    p=0.5
)
```

نتیجه ثبت‌شده برای آن اجرای augmentation برابر 95.84% بود.

### Final Normalization and Augmentation

در اجرای نهایی بهتر، augmentation ملایم به همراه ImageNet normalization استفاده شد.

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

برای Validation از augmentation تصادفی استفاده نشد؛ فقط Resize، ToTensor و normalization اعمال شدند.

بهترین نتیجه ثبت‌شده:

```text
Best Validation Accuracy: 97.40%
```

---

## 10. Final Validation Results

نتایج گزارش‌شده برای بهترین اجرای Validation مدل ResNet18:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| ambulance | 1.0000 | 1.0000 | 1.0000 | 43 |
| autobus | 0.9804 | 1.0000 | 0.9901 | 50 |
| kamyun | 0.9730 | 0.9000 | 0.9351 | 40 |
| kamyunet | 0.9355 | 1.0000 | 0.9667 | 58 |
| minibus | 1.0000 | 1.0000 | 1.0000 | 45 |
| savari | 0.9245 | 0.9800 | 0.9515 | 50 |
| taxi | 1.0000 | 0.9796 | 0.9897 | 49 |
| vanet | 1.0000 | 0.9200 | 0.9583 | 50 |
| **Macro avg** | **0.9767** | **0.9724** | **0.9739** | **385** |
| **Weighted avg** | **0.9751** | **0.9740** | **0.9739** | **385** |

### Validation Confusion Matrix

ترتیب کلاس‌ها در ماتریس:

```text
ambulance, autobus, kamyun, kamyunet, minibus, savari, taxi, vanet
```

ماتریس ثبت‌شده:

```text
[[43,  0,  0,  0,  0,  0,  0,  0],
 [ 0, 50,  0,  0,  0,  0,  0,  0],
 [ 0,  0, 36,  4,  0,  0,  0,  0],
 [ 0,  0,  0, 58,  0,  0,  0,  0],
 [ 0,  0,  0,  0, 45,  0,  0,  0],
 [ 0,  1,  0,  0,  0, 49,  0,  0],
 [ 0,  0,  0,  0,  0,  1, 48,  0],
 [ 0,  0,  1,  0,  0,  3,  0, 46]]
```

در این اجرا، بخشی از خطاها به اشتباه گرفتن `kamyun` با `kamyunet` و اشتباهات محدود بین `savari`، `autobus`، `taxi` و `vanet` مربوط بودند.

---

## 11. Confidence Threshold

برای بررسی اعتماد مدل به پیش‌بینی‌ها، چند threshold روی Validation مقایسه شدند. اگر confidence یک تصویر از threshold کمتر بود، تصویر برای بررسی بیشتر علامت‌گذاری شد.

| Threshold | Review Rate | Coverage | Accepted Accuracy | Error Capture |
|---:|---:|---:|---:|---:|
| 0.50 | 1.30% | 98.70% | 97.63% | 10% |
| 0.60 | 2.34% | 97.66% | 98.40% | 40% |
| 0.70 | 5.71% | 94.29% | 99.17% | 70% |
| 0.80 | 8.83% | 91.17% | 99.72% | 90% |
| 0.90 | 16.10% | 83.90% | 100.00% | 100% |

Threshold استفاده‌شده در checkpoint نهایی:

```text
selected_threshold = 0.70
```

قانون تصمیم‌گیری:

```python
needs_review = confidence < threshold
```

در نتیجه، مدل علاوه بر کلاس پیش‌بینی‌شده و confidence، مشخص می‌کند که آیا تصویر نیاز به بررسی بیشتر دارد یا خیر.

---

## 12. Final Test Evaluation

در آخرین Notebook ارزیابی، مدل نهایی روی 400 تصویر Test ارزیابی شد.

```text
Test Loss:           0.1200
Test Accuracy:       96.75%
Macro Precision:     96.94%
Macro Recall:        96.51%
Macro F1:            96.65%
Weighted Precision:  96.83%
Weighted Recall:     96.75%
Weighted F1:         96.73%
```

### Per-Class Test Metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| ambulance | 0.9592 | 0.9400 | 0.9495 | 50 |
| autobus | 0.9615 | 1.0000 | 0.9804 | 50 |
| kamyun | 1.0000 | 0.8780 | 0.9351 | 41 |
| kamyunet | 0.9508 | 0.9831 | 0.9667 | 59 |
| minibus | 0.9800 | 0.9800 | 0.9800 | 50 |
| savari | 0.9434 | 1.0000 | 0.9709 | 50 |
| taxi | 1.0000 | 0.9800 | 0.9899 | 50 |
| vanet | 0.9600 | 0.9600 | 0.9600 | 50 |
| **Macro avg** | **0.9694** | **0.9651** | **0.9665** | **400** |
| **Weighted avg** | **0.9683** | **0.9675** | **0.9673** | **400** |

### Test Confusion Matrix

ترتیب کلاس‌ها:

```text
ambulance, autobus, kamyun, kamyunet, minibus, savari, taxi, vanet
```

ماتریس اغتشاش ثبت‌شده:

```text
[[47,  0,  0,  1,  0,  0,  0,  2],
 [ 0, 50,  0,  0,  0,  0,  0,  0],
 [ 1,  2, 36,  1,  1,  0,  0,  0],
 [ 0,  0,  0, 58,  0,  1,  0,  0],
 [ 0,  0,  0,  1, 49,  0,  0,  0],
 [ 0,  0,  0,  0,  0, 50,  0,  0],
 [ 0,  0,  0,  0,  0,  1, 49,  0],
 [ 1,  0,  0,  0,  0,  1,  0, 48]]
```

در این ارزیابی، کمترین Recall مربوط به کلاس `kamyun` با مقدار 87.80% بود. تعدادی از خطاهای مدل نیز بین کلاس‌های `kamyun`، `kamyunet` و کلاس‌های نزدیک به آن‌ها رخ دادند.

---

## 13. Unclean / Neysan Analysis

در Notebook ارزیابی، 250 تصویر از پوشه `unclean - merge/neysan` جداگانه بررسی شدند.

### Prediction Distribution

| Predicted Class | Count | Percentage |
|---|---:|---:|
| vanet | 248 | 99.2% |
| ambulance | 1 | 0.4% |
| kamyun | 1 | 0.4% |
| **Total** | **250** | **100%** |

### Needs Review

با threshold برابر 0.70:

```text
needs_review = false: 245
needs_review = true:    5
Review rate:          2.00%
```

از آنجا که `neysan` یکی از هشت کلاس آموزشی مدل نیست، این نتیجه توزیع پیش‌بینی‌های مدل روی این پوشه را نشان می‌دهد و به‌تنهایی معیار accuracy برای یک کلاس نهم محسوب نمی‌شود.

---

## 14. Final Model and Checkpoint

مدل نهایی بر پایه ResNet18 با وزن‌های pretrained روی ImageNet ساخته شد.

```text
Architecture:       ResNet18
Pretrained weights: ImageNet
Trainable layers:   layer3, layer4, fc
Loss:               CrossEntropyLoss
Optimizer:          Adam
Learning rate:      2e-5
Epochs:             15
Input size:         224 × 224
Confidence threshold: 0.70
```

Checkpoint ثبت‌شده:

```text
models/resnet18_final_with_metadata.pth
```

این checkpoint شامل اطلاعات زیر است:

```text
model_state_dict
architecture
class_mapping
transforms
selected_threshold
seed
configuration
```

فایل‌های حجیم مدل به‌صورت محلی نگهداری می‌شوند و با `.gitignore` از ثبت در Git کنار گذاشته شده‌اند.

---

## 15. Prediction Pipeline

اسکریپت پیش‌بینی در فایل زیر پیاده‌سازی شده است:

```text
src/predict.py
```

این اسکریپت:

1. checkpoint نهایی را بارگذاری می‌کند.
2. نگاشت کلاس‌ها و threshold را از metadata می‌خواند.
3. مدل ResNet18 را می‌سازد و وزن‌ها را بارگذاری می‌کند.
4. تصویر ورودی را به RGB تبدیل می‌کند.
5. Resize و ImageNet normalization را اعمال می‌کند.
6. logits مدل را با Softmax به probability تبدیل می‌کند.
7. کلاس دارای بیشترین probability را انتخاب می‌کند.
8. با threshold ذخیره‌شده، مقدار `needs_review` را تعیین می‌کند.
9. نتیجه را در قالب JSON چاپ می‌کند.

### Run

دستور زیر را از ریشه پروژه اجرا کن:

```powershell
python src\predict.py --image "path\to\image.jpg"
```

### Output Format

```json
{
  "predicted_class": "taxi",
  "confidence": 0.87,
  "probabilities": {
    "ambulance": 0.01,
    "autobus": 0.02,
    "kamyun": 0.03,
    "kamyunet": 0.02,
    "minibus": 0.01,
    "savari": 0.02,
    "taxi": 0.87,
    "vanet": 0.02
  },
  "needs_review": false
}
```

مقادیر بالا صرفاً نمونه‌ای برای نمایش قالب خروجی هستند. پیش‌بینی و probabilityهای واقعی به تصویر ورودی بستگی دارند.

در یکی از اجراهای ثبت‌شده، اسکریپت روی یک تصویر خارجی اجرا شد و پیش‌بینی `taxi` با confidence برابر 0.3348 تولید کرد؛ به دلیل پایین بودن confidence از threshold، مقدار `needs_review` برابر `true` بود.

---

## 16. Repository Structure

ساختار منطقی فایل‌های پروژه:

```text
Traffic-Vehicle-Classification-project-2/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── Project-Definition2-traffic-vehicle.ipynb
│   ├── data_audit_01.ipynb
│   ├── data_cleaning_02.ipynb
│   ├── dataloader_03.ipynb
│   ├── model_training.ipynb
│   └── Evaluation-traffic-vehicle.ipynb
├── src/
│   ├── data_loader.py
│   └── predict.py
├── models/
│   └── resnet18_final_with_metadata.pth
└── data/
    ├── train - merge/
    ├── train - merge - clean/
    ├── test/
    └── unclean - merge/
```

نام واقعی برخی فایل‌ها ممکن است در نسخه محلی مخزن شامل شماره نسخه یا پسوند متفاوت باشد. پوشه‌های داده و checkpointهای حجیم به‌صورت محلی نگهداری می‌شوند.

---

## 17. Requirements

کتابخانه‌های ثبت‌شده در `requirements.txt`:

```text
torch
torchvision
numpy
pandas
scikit-learn
Pillow
ImageHash
jupyter
ipykernel
matplotlib
```

---

## Conclusion

در این پروژه ابتدا داده‌ها ممیزی و پاک‌سازی شدند، سپس Train و Validation ساخته شدند. پس از پیاده‌سازی CNN پایه و اجرای آزمایش‌های Dropout، Pooling، Weight Decay، Scheduler، Balanced Batches و مقایسه lossها، چند معماری pretrained بررسی شدند.

در ادامه، ResNet18 با وزن‌های pretrained روی ImageNet و fine-tuning لایه‌های `layer3`، `layer4` و `fc` آموزش داده شد. ترکیب augmentation ملایم و ImageNet normalization بهترین نتیجه ثبت‌شده Validation را ارائه کرد.

نتایج نهایی ثبت‌شده:

```text
Best Validation Accuracy: 97.40%
Test Accuracy:            96.75%
Test Macro F1:             96.65%
Selected Threshold:        0.70
```

علاوه بر طبقه‌بندی هشت کلاس، پروژه شامل تحلیل confidence، تعیین نمونه‌های نیازمند بررسی، تحلیل جداگانه تصاویر Neysan و اسکریپت `predict.py` با خروجی JSON است.