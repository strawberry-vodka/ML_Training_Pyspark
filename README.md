from pymongo import MongoClient
import pandas as pd
import random

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["ab_test_db"]
collection = db["user_assignments"]

def get_user_flag(user_id: str, test_id: str, weight_1: float) -> int:
    # Check if user already assigned for the given test_id
    existing = collection.find_one({"user_id": user_id, "test_id": test_id})
    if existing:
        return existing["flag"]
    
    # Use random.choices to assign 0 or 1 based on weight
    weights = [1 - weight_1, weight_1]  # 0 -> control, 1 -> treatment
    flag = random.choices([0, 1], weights=weights, k=1)[0]

    # Save assignment in MongoDB
    collection.insert_one({
        "user_id": user_id,
        "test_id": test_id,
        "flag": flag
    })
    
    return flag

def assign_flags_to_users(df_users: pd.DataFrame, test_id: str, weight_1: float = 0.5) -> pd.DataFrame:
    df_users["flag"] = df_users["user_id"].apply(lambda uid: get_user_flag(uid, test_id, weight_1))
    return df_users

# Example usage
if __name__ == "__main__":
    # Simulate user dataset
    data = {
        "user_id": [f"user_{i}" for i in range(1, 11)]
    }
    df_users = pd.DataFrame(data)
    
    test_id = "ab_test_june_2025"
    weight_1 = 0.7  # 70% chance of being in group 1

    df_result = assign_flags_to_users(df_users, test_id, weight_1)
    print(df_result)
