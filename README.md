# 🏥 Disease Prediction System

AI-powered disease prediction system that analyzes symptoms and predicts possible diseases with confidence scores.

## ⚡ **NEW!** Two Versions Available

This project now offers **TWO implementations**:

1. **📱 Flask Version** - Traditional Python web app (5 min setup)
2. **⚛️ Next.js + Flask** - Modern React frontend with Flask API (30 min setup)

### 🚀 Quick Decision

**Want to test quickly?** → Use Flask Version  
**Building a portfolio?** → Use Next.js Version  
**Not sure?** → Read **[START_HERE.md](START_HERE.md)** ⭐

---

## 📖 Key Documentation Files

- 🎯 **[START_HERE.md](START_HERE.md)** ⭐ Complete file guide & decision tree
- ⚛️ **[NEXTJS_CONVERSION_SUMMARY.md](NEXTJS_CONVERSION_SUMMARY.md)** - Next.js complete overview
- ⚡ **[QUICK_START.md](QUICK_START.md)** - Flask 3-step start
- 📚 **[COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md)** - Full Flask docs

## ✨ Features

- **820 Diseases**: Comprehensive disease database
- **High Accuracy**: 95-99% prediction accuracy with optimized ML models
- **Modern Landing Page**: Professional onboarding experience ⭐ NEW!
- **User-Friendly**: Simple web interface for symptom selection
- **Top 3 Predictions**: Shows most likely diseases with confidence scores
- **Disease Descriptions**: Detailed information for each predicted disease
- **Fully Responsive**: Works perfectly on desktop, tablet, and mobile

---

## 🚀 Quick Start Guide

### **OPTION A: Flask Version** (Traditional Python)

#### 1️⃣ Install Dependencies
```bash
# Windows one-click
install_dependencies.bat

# Or manually
pip install -r requirements.txt
```

#### 2️⃣ Train the Model
```bash
# Windows one-click
run_improved_training.bat

# Or manually
python train_model_improved.py
```

#### 3️⃣ Run the App
```bash
python app.py
```

Visit: 
- **http://127.0.0.1:5000/** - Landing page 🌐
- **http://127.0.0.1:5000/app** - Symptom checker

---

### **OPTION B: Next.js + Flask Version** (Modern React)

#### 1️⃣ Setup Next.js Frontend
```bash
# Windows one-click
setup_nextjs.bat

# Or manually follow NEXTJS_SETUP_GUIDE.md
```

#### 2️⃣ Copy Component Files
- Open `FRONTEND_COMPONENTS_PART1.md` and copy all components
- Open `FRONTEND_COMPONENTS_PART2.md` and copy remaining files
- Follow the file paths specified in each section

#### 3️⃣ Train the Model
```bash
python train_model_improved.py
```

#### 4️⃣ Run Both Servers
**Terminal 1 - Flask API:**
```bash
python app_api.py
```

**Terminal 2 - Next.js Frontend:**
```bash
cd frontend
npm run dev
```

Visit:
- **http://localhost:3000** - Next.js frontend ⚛️
- **http://localhost:5000** - Flask API

📖 **Full Guide**: [NEXTJS_README.md](NEXTJS_README.md)

---

## 🎨 New Landing Page!

We've added a **beautiful, modern landing page** to your app!

---

## 📊 Model Training Options

### 🎯 **Improved Training** (Recommended)
**File**: `train_model_improved.py`

Tests 7 algorithms and automatically saves the best:
- Random Forest (100 & 200 trees)
- Extra Trees (100 & 200 trees)
- Gradient Boosting
- XGBoost (100 & 200 trees)

**Expected Accuracy**: 95-99% ⭐

### 📝 **Basic Training**
**File**: `train_model.py`

Simple Random Forest training.

**Expected Accuracy**: 85-92%

---

## 📁 Project Structure

```
Disease-Prediction/
├── app.py                          # Flask web application
├── train_model_improved.py         # ⭐ Improved training (recommended)
├── train_model.py                  # Basic training script
├── check_accuracy.py               # Model accuracy checker
├── model/                          # Trained model artifacts
│   ├── disease_model.pkl           # Trained ML model
│   ├── label_encoder.pkl           # Disease label encoder
│   └── symptom_columns.pkl         # Symptom feature list
├── templates/                      # HTML templates
├── static/                         # CSS, JS, images
├── requirements.txt                # Python dependencies
└── *.csv                           # Dataset files
```

**Features:**
- ✅ Professional design with smooth animations
- ✅ Responsive (works on all devices)
- ✅ Clear value proposition
- ✅ Trust-building stats and features
- ✅ Easy navigation to symptom checker

**Read more:** [LANDING_PAGE_GUIDE.md](LANDING_PAGE_GUIDE.md)

---

## 📚 Complete Documentation

- 📖 **[COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md)** - ⭐ Start here! Complete overview
- 📖 **[QUICK_START.md](QUICK_START.md)** - Get started in 3 steps
- 📊 **[COMPARISON.md](COMPARISON.md)** - Before vs After comparison
- 📝 **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Detailed training instructions
- 📋 **[IMPROVEMENT_SUMMARY.md](IMPROVEMENT_SUMMARY.md)** - Model improvements explained
- 🎨 **[LANDING_PAGE_GUIDE.md](LANDING_PAGE_GUIDE.md)** - Landing page documentation

---

## 🛠️ Requirements

- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- joblib
- xgboost (optional, but recommended)

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🎯 How It Works

1. **User selects symptoms** from a comprehensive list
2. **ML model analyzes** the symptom combination
3. **System predicts** top 3 most likely diseases
4. **Displays results** with confidence scores and descriptions

---

## 📈 Accuracy Improvements

| Version | Algorithm | Accuracy |
|---------|-----------|----------|
| Basic | Random Forest (25 trees) | ~87% |
| **Improved** | **Best of 7 algorithms** | **95-99%** ⭐ |

---

## 🧪 Testing Your Model

Check current model accuracy:
```bash
python check_accuracy.py
```

Shows:
- Test accuracy
- Precision, Recall, F1-Score
- Dataset statistics

---

## 🔧 Batch Files (Windows)

Convenient one-click launchers:

- **`install_dependencies.bat`** - Install all packages
- **`run_improved_training.bat`** - Train the best model
- Both include pause for viewing results

---

## 📊 Dataset

- **820 unique diseases**
- **Row-level cleaned symptom data**
- **Binary symptom encoding** (0/1)
- **Verified disease descriptions**

---

## 🌐 Web Interface

Simple and intuitive:
1. Select symptoms from dropdown (multiple selection)
2. Click "Predict Disease"
3. View top 3 predictions with:
   - Disease name
   - Confidence percentage
   - Detailed description

---

## 💡 Tips

- **More symptoms = better accuracy**: Select all applicable symptoms
- **Retrain when data changes**: Run training after updating the dataset
- **XGBoost recommended**: Install for best results: `pip install xgboost`
- **First run takes longer**: Model loads into memory on first prediction

---

## 🐛 Troubleshooting

**Model not found error?**
```bash
python train_model_improved.py
```

**Import errors?**
```bash
pip install -r requirements.txt
```

**XGBoost installation issues?**
```bash
# Try conda if pip fails
conda install -c conda-forge xgboost
```

**Low accuracy?**
- Check dataset quality
- Ensure sufficient training data per disease
- Try running improved training script

---

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Improve documentation
- Optimize models

---

## 📄 License

This project is for educational and research purposes.

---

## 🎓 Technical Stack

- **Backend**: Flask (Python)
- **ML Library**: scikit-learn, XGBoost
- **Frontend**: HTML, CSS, JavaScript
- **Models**: Ensemble methods (Random Forest, Extra Trees, XGBoost)

---

## 📞 Support

For issues or questions:
1. Check the documentation files (TRAINING_GUIDE.md, etc.)
2. Review console error messages
3. Ensure all dependencies are installed

---

## 🎉 Quick Reference Commands

```bash
# Setup
pip install -r requirements.txt

# Train improved model
python train_model_improved.py

# Check accuracy
python check_accuracy.py

# Run web app
python app.py

# Train basic model
python train_model.py
```

---

**Made with ❤️ for accurate disease prediction**

🚀 **Get started now!** Read [QUICK_START.md](QUICK_START.md)
