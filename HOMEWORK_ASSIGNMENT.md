# 🎯 Machine Learning Homework Assignment

## **Mission: Build Your Own Passenger Survival Analyzer**

### 📋 **Assignment Overview**
You've learned how the Titanic predictor works. Now it's time to create your own version with some exciting twists! This homework will test your understanding while keeping things fun and manageable.

---

## 🎮 **Your Tasks (Choose 3 out of 5)**

### **Task 1: The Feature Detective** 🕵️‍♀️
**Goal:** Discover which features are most important by experimenting

**What to do:**
1. Run the original `titanic_predictor.py` and note the accuracy
2. Create a new version that removes ONE feature at a time
3. Test these scenarios:
   - Remove `Sex` (gender)
   - Remove `Pclass` (passenger class)
   - Remove `Age`
   - Remove `FamilySize`

**Questions to answer:**
- Which feature removal hurt accuracy the most?
- Which feature seems least important?
- Can you explain WHY gender or class might be so important?

**Expected output:** A simple table showing accuracy with/without each feature

---

### **Task 2: The Age Group Explorer** 👶👴
**Goal:** Create better age categories and see if it improves predictions

**What to do:**
1. Instead of the current age groups, create these new categories:
   - `Baby` (0-2 years)
   - `Child` (3-12 years)
   - `Teen` (13-19 years)
   - `Adult` (20-59 years)
   - `Senior` (60+ years)

2. Replace the existing age groups with your new ones
3. Compare the accuracy before and after

**Questions to answer:**
- Did your new age groups improve accuracy?
- Which age group had the highest survival rate?
- Why might babies have different survival rates than other children?

**Bonus:** Create a bar chart showing survival rates by your new age groups

---

### **Task 3: The Fare Investigator** 💰
**Goal:** Explore how ticket prices affected survival chances

**What to do:**
1. Create a new feature called `FarePerPerson`:
   ```python
   df['FarePerPerson'] = df['Fare'] / df['FamilySize']
   ```

2. Create fare categories:
   - `Free` (fare = 0)
   - `Cheap` (fare: 0-10)
   - `Moderate` (fare: 10-50)
   - `Expensive` (fare: 50+)

3. Add these features to your model and test accuracy

**Questions to answer:**
- Does fare per person predict survival better than total fare?
- Were expensive tickets worth it for survival?
- Why might some passengers have free tickets?

---

### **Task 4: The Title Extractor** 🎭
**Goal:** Extract titles from passenger names and use them as features

**What to do:**
1. Extract titles from the `Name` column:
   ```python
   df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
   ```

2. Group rare titles together:
   ```python
   # Keep common titles, group others as 'Other'
   common_titles = ['Mr', 'Mrs', 'Miss', 'Master']
   df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Other')
   ```

3. Convert to dummy variables and add to your model

**Questions to answer:**
- Which titles had the highest survival rates?
- Did adding titles improve your model's accuracy?
- What does "Master" mean, and why might it be important?

**Hint:** Master was used for young boys, Mrs for married women, Miss for unmarried women

---

### **Task 5: The Model Experimenter** 🧪
**Goal:** Try a new machine learning algorithm

**What to do:**
1. Add a new model to test - Support Vector Machine (SVM):
   ```python
   from sklearn.svm import SVC
   
   # Add to your models dictionary
   'SVM': SVC(probability=True, random_state=42)
   ```

2. Compare all four models (Logistic Regression, Decision Tree, Random Forest, SVM)
3. Create a simple bar chart showing their accuracies

**Questions to answer:**
- Which model performed best on your data?
- Which model was fastest to train?
- If you had to explain predictions to a non-technical person, which model would you choose and why?

---