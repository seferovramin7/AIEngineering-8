#!/usr/bin/env python3
"""
🚢 Interactive Titanic Survival Predictor Demo
A fun, interactive way to explore the Titanic dataset and make predictions!
"""

import sys
from titanic_predictor import TitanicPredictor

def get_user_input():
    """Get passenger details from user input"""
    print("\n🎭 CREATE YOUR PASSENGER")
    print("=" * 40)
    
    # Get passenger details
    print("Let's create a passenger and see if they would have survived the Titanic!")
    
    # Passenger class
    print("\n1️⃣ Passenger Class:")
    print("   1 = First Class (luxury)")
    print("   2 = Second Class (middle)")
    print("   3 = Third Class (economy)")
    while True:
        try:
            pclass = int(input("Enter passenger class (1-3): "))
            if pclass in [1, 2, 3]:
                break
            print("Please enter 1, 2, or 3")
        except ValueError:
            print("Please enter a valid number")
    
    # Gender
    print("\n2️⃣ Gender:")
    while True:
        sex = input("Enter gender (male/female): ").lower()
        if sex in ['male', 'female']:
            sex_numeric = 0 if sex == 'male' else 1
            break
        print("Please enter 'male' or 'female'")
    
    # Age
    print("\n3️⃣ Age:")
    while True:
        try:
            age = float(input("Enter age: "))
            if 0 <= age <= 100:
                break
            print("Please enter a realistic age (0-100)")
        except ValueError:
            print("Please enter a valid number")
    
    # Family details
    print("\n4️⃣ Family Details:")
    while True:
        try:
            sibsp = int(input("Number of siblings/spouses aboard: "))
            if sibsp >= 0:
                break
            print("Please enter a non-negative number")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        try:
            parch = int(input("Number of parents/children aboard: "))
            if parch >= 0:
                break
            print("Please enter a non-negative number")
        except ValueError:
            print("Please enter a valid number")
    
    # Fare
    print("\n5️⃣ Ticket Fare:")
    print("   Typical fares: 1st class: $50-500, 2nd class: $10-50, 3rd class: $5-15")
    while True:
        try:
            fare = float(input("Enter ticket fare ($): "))
            if fare >= 0:
                break
            print("Please enter a non-negative fare")
        except ValueError:
            print("Please enter a valid number")
    
    # Embarkation port
    print("\n6️⃣ Embarkation Port:")
    print("   C = Cherbourg, France")
    print("   Q = Queenstown, Ireland")
    print("   S = Southampton, England")
    while True:
        embarked = input("Enter embarkation port (C/Q/S): ").upper()
        if embarked in ['C', 'Q', 'S']:
            break
        print("Please enter C, Q, or S")
    
    # Calculate derived features
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    # Create age group
    if age < 12:
        age_group = [1, 0, 0, 0, 0]  # Child
    elif age < 18:
        age_group = [0, 1, 0, 0, 0]  # Teen
    elif age < 35:
        age_group = [0, 0, 1, 0, 0]  # Adult
    elif age < 60:
        age_group = [0, 0, 0, 1, 0]  # Middle
    else:
        age_group = [0, 0, 0, 0, 1]  # Senior
    
    # Create fare group (simplified)
    if fare < 10:
        fare_group = [1, 0, 0, 0]  # Low
    elif fare < 25:
        fare_group = [0, 1, 0, 0]  # Medium
    elif fare < 50:
        fare_group = [0, 0, 1, 0]  # High
    else:
        fare_group = [0, 0, 0, 1]  # Very High
    
    # Create embarked one-hot
    embarked_features = [0, 0, 0]
    if embarked == 'C':
        embarked_features[0] = 1
    elif embarked == 'Q':
        embarked_features[1] = 1
    else:  # S
        embarked_features[2] = 1
    
    # Create feature vector
    passenger_data = [
        pclass, sex_numeric, age, sibsp, parch, fare,
        family_size, is_alone
    ] + embarked_features + age_group + fare_group
    
    return passenger_data, {
        'class': pclass,
        'sex': sex,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'fare': fare,
        'embarked': embarked,
        'family_size': family_size,
        'is_alone': is_alone
    }

def display_passenger_summary(passenger_info):
    """Display a summary of the passenger"""
    print("\n👤 PASSENGER SUMMARY")
    print("=" * 40)
    
    class_names = {1: "First Class", 2: "Second Class", 3: "Third Class"}
    embarked_names = {'C': "Cherbourg, France", 'Q': "Queenstown, Ireland", 'S': "Southampton, England"}
    
    print(f"🎫 Class: {class_names[passenger_info['class']]}")
    print(f"👤 Gender: {passenger_info['sex'].title()}")
    print(f"🎂 Age: {passenger_info['age']}")
    print(f"👨‍👩‍👧‍👦 Family Size: {passenger_info['family_size']}")
    if passenger_info['is_alone']:
        print(f"🚶 Traveling: Alone")
    else:
        print(f"🚶 Traveling: With {passenger_info['family_size']-1} family members")
    print(f"💰 Fare: ${passenger_info['fare']:.2f}")
    print(f"🚢 Embarked: {embarked_names[passenger_info['embarked']]}")

def main():
    """Main interactive demo function"""
    print("🚢" * 25)
    print("🚢 INTERACTIVE TITANIC SURVIVAL PREDICTOR")
    print("🚢" * 25)
    
    print("\nWelcome to the interactive Titanic survival predictor!")
    print("This demo will let you create a passenger and predict their survival.")
    print("\nFirst, let's train our machine learning model...")
    
    # Create and train the predictor
    predictor = TitanicPredictor()
    predictor.load_data()
    predictor.engineer_features()
    predictor.prepare_features()
    predictor.build_models()
    
    print("\n✅ Model trained successfully!")
    
    while True:
        try:
            # Get passenger details
            passenger_data, passenger_info = get_user_input()
            
            # Display passenger summary
            display_passenger_summary(passenger_info)
            
            # Make prediction
            print("\n🔮 SURVIVAL PREDICTION")
            print("=" * 40)
            
            prediction, probability = predictor.predict_survival(passenger_data)
            
            if prediction == 1:
                print("🎉 PREDICTION: This passenger would likely have SURVIVED!")
                print(f"✨ Survival probability: {probability[1]:.1%}")
            else:
                print("😔 PREDICTION: This passenger would likely have NOT SURVIVED")
                print(f"💔 Survival probability: {probability[1]:.1%}")
            
            # Explain the prediction
            print(f"\n🧠 EXPLANATION:")
            if passenger_info['sex'] == 'female':
                print("   ✅ Being female significantly increased survival chances")
            else:
                print("   ❌ Being male decreased survival chances")
            
            if passenger_info['class'] == 1:
                print("   ✅ First class passengers had better survival rates")
            elif passenger_info['class'] == 2:
                print("   ⚠️ Second class passengers had moderate survival rates")
            else:
                print("   ❌ Third class passengers had lower survival rates")
            
            if passenger_info['age'] < 16:
                print("   ✅ Children had higher survival rates")
            elif passenger_info['age'] > 60:
                print("   ❌ Elderly passengers had lower survival rates")
            
            if passenger_info['family_size'] == 1:
                print("   ❌ Traveling alone decreased survival chances")
            elif passenger_info['family_size'] > 4:
                print("   ❌ Very large families had difficulty surviving together")
            else:
                print("   ✅ Traveling with a small family helped survival")
            
            # Ask if user wants to try again
            print("\n" + "="*50)
            again = input("Would you like to try another passenger? (y/n): ").lower()
            if again not in ['y', 'yes']:
                break
                
        except KeyboardInterrupt:
            print("\n\nThanks for using the Titanic Survival Predictor!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Let's try again...")
            continue
    
    print("\n🚢 Thank you for exploring the Titanic dataset!")
    print("🎓 You've learned about:")
    print("   • Feature engineering and data preprocessing")
    print("   • Machine learning model training")
    print("   • Making predictions with real data")
    print("   • Understanding model explanations")
    print("\n🚀 Keep exploring and learning!")

if __name__ == "__main__":
    main() 