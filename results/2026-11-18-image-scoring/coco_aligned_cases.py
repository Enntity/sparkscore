"""Real-photo gate cases: 24 COCO val2017 images x 6 general questions (one request per image)."""
import glob, os
QUESTIONS = {
    "person": {"type": "noul", "instructions": "Is there at least one person in the picture?"},
    "indoors": {"type": "noul", "instructions": "Was this photo taken indoors?"},
    "subject": {"type": "choice", "instructions": "What is the main subject of the photo?",
                "criteria": {"People": None, "An animal": None, "A vehicle": None, "Food": None,
                             "Furniture or a room": None, "A landscape or street scene": None, "Sports equipment": None}},
    "count": {"type": "choice", "instructions": "How many people are visible?",
              "criteria": {"None": None, "One": None, "Two or three": None, "Four or more": None}},
    "light": {"type": "noul", "instructions": "Is the scene lit by daylight?"},
    "sharp": {"type": "score", "instructions": "How sharp and well exposed is the photo?",
              "criteria": ["Very poor", "Poor", "Fair", "Good", "Excellent"]},
}
STATES = [None, "Photo uploaded to a listing; the reviewer flags anything unusual."]
def cases():
    files = sorted(os.path.basename(f) for f in glob.glob("/w/d3work/img/coco_aligned/*.png"))
    return [{"id": f"coco-{f[:-4].lstrip('0')}", "image": f"coco_aligned/{f}", "state": STATES[i % 2], "questions": QUESTIONS}
            for i, f in enumerate(files)]
