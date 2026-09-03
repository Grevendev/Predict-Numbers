# 📱 Handwritten Digit Classifier – MNIST

En enkel AI-applikation byggd med **Python, Machine Learning och Streamlit** som klassificerar handskrivna siffror från **0–9**.

Användaren kan ladda upp en bild av en handskriven siffra och den tränade modellen försöker identifiera vilken siffra bilden föreställer. Applikationen visar både den uppladdade bilden, den förbehandlade bilden och modellens prediktion tillsammans med en confidence score.

## 🚀 Demo

Applikationen låter användaren:

1. Ladda upp en bild av en handskriven siffra.
2. Konvertera bilden till gråskala.
3. Automatiskt invertera bilden om bakgrunden är ljus.
4. Skala ner bilden till **28 × 28 pixlar**.
5. Normalisera pixelvärdena till intervallet **0–1**.
6. Skicka bilden till den tränade ML-modellen.
7. Visa modellens prediktion och confidence.

Exempel:

```text
Modellen gissar att siffran är: 7
Förtroende (Confidence): 98.4%
```

---

## 🧠 Om MNIST

**MNIST** är ett klassiskt dataset inom Machine Learning och Computer Vision som innehåller bilder av handskrivna siffror från 0 till 9.

Varje bild är:

* 28 × 28 pixlar
* Gråskala
* 784 pixelvärden totalt
* Tillhör en av 10 klasser: `0–9`

Modellen tränas på MNIST-data och används sedan för att klassificera nya bilder.

---

## ⚙️ Preprocessing

En viktig del av projektet är bildens preprocessing.

MNIST-bilder har normalt:

* Svart bakgrund
* Vit/grå siffra
* Storlek: 28 × 28 pixlar

Uppladdade bilder kan däremot se helt annorlunda ut. Därför genomför applikationen flera steg innan bilden skickas till modellen.

### 1. Konvertering till gråskala

```python
img_gray = image.convert("L")
```

Bilden konverteras till grayscale för att matcha MNIST-formatet.

### 2. Automatisk invertering

Applikationen kontrollerar bildens genomsnittliga pixelvärde.

Om bilden har en ljus bakgrund antas det att siffran är mörk och bilden inverteras:

```python
if img_array_check.mean() > 127:
    img_gray = ImageOps.invert(img_gray)
```

Detta gör att bilden bättre liknar MNIST-formatet.

### 3. Resize till 28 × 28

```python
img_resized = img_gray.resize(
    (28, 28),
    Image.Resampling.LANCZOS
)
```

Bilden skalas ner till exakt samma dimension som MNIST-bilderna.

### 4. Normalisering

Pixelvärdena ligger normalt mellan `0–255`.

De normaliseras därför till `0–1`:

```python
img_np = np.array(img_resized).reshape(1, -1) / 255.0
```

Bilden omvandlas samtidigt från:

```text
28 × 28
```

till:

```text
1 × 784
```

vilket motsvarar de 784 features som modellen tränats på.

---

## 🤖 Machine Learning Model

Den tränade modellen sparas som:

```text
mnist_model.pkl
```

Modellen laddas med `joblib`:

```python
@st.cache_resource
def load_model():
    return joblib.load("mnist_model.pkl")
```

Streamlit cache används för att undvika att modellen laddas från disk varje gång applikationen körs eller uppdateras.

Modellen används sedan för både klassificering och sannolikheter:

```python
prediction = model.predict(img_np)

probabilities = model.predict_proba(img_np)
```

Confidence beräknas genom att ta den högsta klass-sannolikheten:

```python
confidence = np.max(probabilities) * 100
```

---

## 🖥️ Applikationens flöde

```text
             ┌──────────────────┐
             │  Ladda upp bild  │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │    Gråskala      │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Invertera behov  │
             │      vid behov   │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Resize 28 × 28   │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Normalisering    │
             │      0–1         │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │  ML Prediction   │
             └────────┬─────────┘
                      │
             ┌────────┴─────────┐
             ▼                  ▼
       ┌───────────┐      ┌────────────┐
       │ Prediktion│      │ Confidence │
       └───────────┘      └────────────┘
```

---

## 🛠️ Tekniker

Projektet använder bland annat:

* **Python**
* **Machine Learning**
* **MNIST**
* **Streamlit**
* **NumPy**
* **Pillow (PIL)**
* **Joblib**
* **scikit-learn** *(för den tränade modellen)*

### Python packages

```text
streamlit
numpy
pillow
joblib
scikit-learn
```

---

## 📁 Projektstruktur

```text
mnist-digit-classifier/
│
├── app.py
├── mnist_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

### Filer

| Fil                | Beskrivning                         |
| ------------------ | ----------------------------------- |
| `app.py`           | Streamlit-applikationen             |
| `mnist_model.pkl`  | Tränad Machine Learning-modell      |
| `requirements.txt` | Projektets Python-dependencies      |
| `README.md`        | Dokumentation                       |
| `.gitignore`       | Filer som inte ska versionshanteras |

---

## ▶️ Installation

### 1. Klona repositoryt

```bash
git clone <repository-url>
cd mnist-digit-classifier
```

### 2. Skapa en virtuell miljö

Windows:

```bash
python -m venv .venv
```

Aktivera miljön:

```bash
source .venv/Scripts/activate
```

eller i PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Installera dependencies

```bash
pip install -r requirements.txt
```

### 4. Starta applikationen

```bash
streamlit run app.py
```

Applikationen blir därefter tillgänglig lokalt via Streamlit.

---

## 📷 Användning

När applikationen startats:

1. Klicka på **"Välj en bildfil"**.
2. Ladda upp en `.jpg`, `.jpeg` eller `.png`.
3. Applikationen preprocessar bilden.
4. Den förbehandlade bilden visas.
5. Modellen klassificerar siffran.
6. Prediktionen och confidence visas i gränssnittet.

För bästa resultat bör siffran:

* Vara tydligt synlig.
* Vara relativt centrerad.
* Ha bra kontrast mot bakgrunden.
* Inte innehålla för mycket bakgrund eller brus.

---

## 🎯 Syfte med projektet

Projektets syfte är att demonstrera ett komplett Machine Learning-flöde från **input till prediction**.

Projektet visar bland annat:

* Hur en tränad ML-modell kan sparas och laddas.
* Hur preprocessing påverkar Machine Learning-modeller.
* Hur bilddata kan konverteras till numeriska features.
* Hur en modell kan användas för inference.
* Hur sannolikheter kan användas för att beräkna confidence.
* Hur en ML-modell kan integreras i en användarvänlig applikation med Streamlit.

---

## 🔮 Möjliga förbättringar

Projektet kan vidareutvecklas på flera sätt:

* [ ] Använda en CNN istället för en traditionell ML-modell.
* [ ] Bättre bildsegmentering och centrering.
* [ ] Automatisk beskärning av siffran.
* [ ] Visualisera sannolikheten för samtliga siffror `0–9`.
* [ ] Lägga till möjlighet att rita en siffra direkt i webbläsaren.
* [ ] Förbättra preprocessing för fotografier.
* [ ] Distribuera applikationen online.
* [ ] Jämföra flera olika ML-modeller.
* [ ] Mäta modellens accuracy, precision, recall och F1-score.

---

## 📚 Vad jag lärde mig

Projektet ger praktisk erfarenhet av hela kedjan:

```text
Dataset
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Inference
   ↓
Web Application
```

Det visar hur en Machine Learning-modell kan gå från ett träningsdataset till en faktisk applikation där en användare kan interagera med modellen.

---

## 👨‍💻 Author

**Grevendev**

Projektet är skapat som en del av mitt arbete med **Python, Machine Learning och AI-utveckling**.
