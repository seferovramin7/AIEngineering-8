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

## 📝 **Submission Requirements**

### **What to Submit:**
1. **Your modified Python script** (name it `homework_solution.py`)
2. **A short report** (1-2 pages) answering the questions from your chosen tasks
3. **Screenshots** of any charts or outputs you created

### **Report Format:**
```
# My Machine Learning Homework Report

## Tasks Completed: [List which 3 tasks you chose]

## Task X: [Task Name]
### What I Did:
- [Brief description]

### Results:
- [Key findings]

### Answers to Questions:
1. [Answer to question 1]
2. [Answer to question 2]
...

## What I Learned:
[2-3 sentences about what surprised you or what you found interesting]

## Challenges:
[Any difficulties you faced and how you solved them]
```

---

## 🎯 **Grading Criteria**

| Criteria | Points | What We're Looking For |
|----------|--------|----------------------|
| **Code Works** | 40% | Script runs without errors, produces results |
| **Understanding** | 30% | Answers show you understand what's happening |
| **Exploration** | 20% | You tried different approaches, showed curiosity |
| **Presentation** | 10% | Clear writing, organized results |

---

## 💡 **Helpful Tips**

### **Getting Started:**
1. **Copy the original script** - don't modify the working version
2. **Test small changes first** - make one change at a time
3. **Print everything** - use `print()` statements to see what's happening
4. **Save your work often** - make backups of working versions

### **If You Get Stuck:**
1. **Read the error message** - it usually tells you what's wrong
2. **Check your data** - use `df.head()` to see what your data looks like
3. **Start simple** - get basic functionality working first
4. **Ask for help** - but try to solve it yourself first

### **Making It Interesting:**
- **Create charts** - visualizations make findings more compelling
- **Tell a story** - explain what your data reveals about the Titanic
- **Be curious** - ask "what if" questions and test them

---

## 🚀 **Bonus Challenges** (Optional)

### **Easy Bonus (+5 points):**
Create a function that takes passenger details and returns a "survival story":
```python
def survival_story(passenger_info):
    # Return a sentence like:
    # "This 25-year-old woman in first class had an 85% chance of survival because..."
```

### **Medium Bonus (+10 points):**
Create a simple "What-If" analyzer that shows how changing one feature affects survival probability:
```python
# Example: "If this passenger had been in first class instead of third class,
# their survival probability would have increased from 23% to 67%"
```

### **Hard Bonus (+15 points):**
Build a simple web interface using Streamlit where users can input passenger details and get predictions:
```bash
pip install streamlit
streamlit run your_app.py
```

---

## 📚 **Learning Objectives**

By completing this homework, you will:
- ✅ **Understand feature importance** - learn which data matters most
- ✅ **Practice feature engineering** - create meaningful new features
- ✅ **Compare models** - understand different algorithm strengths
- ✅ **Interpret results** - explain what your findings mean
- ✅ **Think like a data scientist** - ask questions and test hypotheses

---

## 🎉 **Final Notes**

Remember, this is about **learning**, not just getting the right answer. If something doesn't work as expected, that's still a valuable learning experience! Document what you tried, what worked, what didn't, and what you learned.

**Most importantly:** Have fun with it! You're working with real historical data and building AI systems. That's pretty amazing! 🚀

---

**Due Date:** [To be filled by instructor]  
**Questions?** [Contact information]

**Good luck, and happy coding!** 🎯 