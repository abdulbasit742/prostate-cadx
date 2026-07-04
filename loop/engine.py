import json
from pathlib import Path
from lib.db import db

class LoopEngine:
    def __init__(self):
        pass

    def get_next_ready_skill(self):
        """
        Picks the highest priority ready pending skill from the DB.
        """
        all_skills = db.get_all_skills()
        done_skills = {s["name"] for s in all_skills if s["status"] == "done"}
        
        ready_skills = []
        for s in all_skills:
            if s["status"] == "pending":
                deps = json.loads(s["deps"])
                if all(d in done_skills for d in deps):
                    ready_skills.append(s)
                    
        if ready_skills:
            # Sort by ID
            ready_skills.sort(key=lambda x: x["id"])
            return ready_skills[0]
        return None
