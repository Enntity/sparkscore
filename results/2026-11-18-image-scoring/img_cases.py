"""Image gate cases: (id, image file, state, questions). Same file is read by the reference and the Atlas client."""
IMAGES = ["01_square_224.png", "03_landscape_512x384.png", "04_wide_640x360.png", "06_wide_1024x576.png",
          "07_hd_1280x720.png", "08_portrait_480x854.png", "09_over_clamp_1600x900.png", "13_gray_224.jpg"]
QUESTIONS = {
    "rect": {"type": "choice", "instructions": "What color is the filled rectangle near the top left of the picture?",
             "criteria": {"White": None, "Black": None, "Red": None, "Green": None, "Blue": None}},
    "oval": {"type": "noul", "instructions": "Is there a dark oval in the picture?"},
    "where": {"type": "choice", "instructions": "Where is the dark oval?",
              "criteria": {"Top left": None, "Top right": None, "Bottom left": None, "Bottom right": None, "There is no oval": None}},
    "kind": {"type": "choice", "instructions": "What kind of picture is this?",
             "criteria": {"A photograph of a natural landscape": None, "A synthetic test pattern with geometric shapes": None,
                          "A portrait of a person": None, "A page of printed text": None, "A city skyline at night": None,
                          "A hand-drawn cartoon": None}},
    "line": {"type": "noul", "instructions": "Does a thin yellow line cross the picture diagonally?"},
}
STATES = [None, "Frame 17 from the quality-control camera on line 3."]
def cases():
    out = []
    for i, image in enumerate(IMAGES):
        state = STATES[i % 2]
        out.append({"id": f"{i:02d}-{image.split('.')[0]}", "image": image, "state": state, "questions": QUESTIONS})
    return out
