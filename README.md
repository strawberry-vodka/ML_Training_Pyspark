import pandas as pd
import random
from pymongo import MongoClient

class ABTestAssigner:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="ab_test_db"):
        self.client = MongoClient(mongo_uri)
        self.collection = self.client[db_name]["user_assignments"]

    def get_existing_flags(self, audience_ids, test_id):
        # Fetch existing records in bulk
        existing_docs = self.collection.find(
            {"user_id": {"$in": audience_ids}, "test_id": test_id},
            {"_id": 0, "user_id": 1, "flag": 1}
        )
        return {doc["user_id"]: doc["flag"] for doc in existing_docs}

    def assign_flags_to_users(self, audienceid_list, test_id: str, weight_1: float):
        """
        Assigns flags (0/1) to users for an A/B test and stores new ones in MongoDB.
        """
        ab_test_dict = {}
        audience_ids = list(set(audienceid_list))  # Ensure unique

        # 1. Fetch already assigned users
        existing_flags = self.get_existing_flags(audience_ids, test_id)
        ab_test_dict.update(existing_flags)

        # 2. Identify unassigned users
        new_users = list(set(audience_ids) - set(existing_flags.keys()))

        # 3. Random assignment for new users
        if new_users:
            new_flags = random.choices([0, 1], weights=[1 - weight_1, weight_1], k=len(new_users))
            new_entries = []
            for user_id, flag in zip(new_users, new_flags):
                ab_test_dict[user_id] = flag
                new_entries.append({
                    "user_id": user_id,
                    "test_id": test_id,
                    "flag": flag
                })

            # 4. Bulk insert new records to MongoDB
            if new_entries:
                self.collection.insert_many(new_entries)

        return ab_test_dict
