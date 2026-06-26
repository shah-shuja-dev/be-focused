## Dataset Recording Guide — Poses in Detail
 
This is the most important part of Day 1. A model is only as good as its dataset.
The CNN learns to distinguish "eyes on screen, head forward" from "looking elsewhere."
 
**Target:** 600–800 images per class minimum.
Each 2-minute video at 1 FPS gives ~120 images. You need 5–6 clips per class.
 
---
 
### Setup before recording
 
- Sit at your laptop exactly where you normally work.
- Don't move the laptop between sessions — consistent camera framing matters.
- Normal room, normal clothes. Don't overthink it.
- Record at different times of day if you can (morning light vs evening lamp).
---
 
### FOCUSED class — what to record
 
All sessions should look like: "I am paying attention to my screen."
 
**Session 1 — Natural working posture (2 min)**
Sit normally and actually use your laptop. Browse something, read text, type.
Don't freeze or hold a rigid pose — natural micro-movements make the dataset robust.
Your head will naturally tilt ±5° as you read. That's fine and wanted.
 
**Session 2 — Leaned back, relaxed (2 min)**
Lean back in your chair, arms resting on the desk or armrests.
Still looking at the screen. This captures "relaxed but working" — should NOT be flagged.
 
**Session 3 — Leaned forward / concentrating (2 min)**
Lean forward slightly, chin slightly down, as if reading something carefully.
You can rest your chin on your hand — as long as your gaze is on screen.
 
**Session 4 — Dim lighting (2 min)**
Turn off the main light. Work under just your screen glow or a side lamp.
Forces the model to generalise beyond bright evenly-lit faces.
 
**Session 5 — Distance variation (2 min)**
First minute: sit closer than normal (~40cm from screen).
Second minute: lean back further than normal (~80–90cm).
Camera-to-face distance changes apparent face size — model needs both.
 
**Session 6 — Visual variation (2 min)**
Wear glasses if you have them. Wear a hoodie or a different collar.
Different backgrounds visible behind you also help.
 
---
 
### NOT FOCUSED class — what to record
 
All sessions should look like: "I am clearly doing something other than looking at my screen."
 
**Session 1 — Phone in lap (2 min)**  ← most important, do 2 clips of this
Hold your phone in your lap and look at it. Head bows down, eyes point downward.
Vary the angle: mild bow (~20° down) and strong bow (~40° down).
This is the single most common real-world violation.
 
**Session 2 — Phone held up at chest (2 min)**
Hold phone at chest/belly height. Head slightly down, eyes angled down-forward.
Different from lap phone because the head pose and eye direction differ.
 
**Session 3 — Looking left (2 min)**
Turn your head left as if someone called your name or you're looking at another window.
Do three sub-angles: gentle (30°), medium (60°), and full turn (90° — profile view).
Hold each for 10–15 seconds, then return to center, then repeat.
 
**Session 4 — Looking right (2 min)**
Same as above, right side. Don't skip — left and right are genuinely different features.
30°, 60°, 90° variation. Return to center between holds.
 
**Session 5 — Looking down at desk (2 min)**
Head tilted forward-down as if reading notes on your desk, checking a book,
or looking at a keyboard. Eyes clearly not on screen.
- Mild tilt: ~20° down (reading something close)
- Strong tilt: ~45° down (reading something flat on the desk)
**Session 6 — Looking up or behind (2 min)**
Head tilted back looking up (like at a TV mounted on the wall).
Then turn to look completely behind you over your shoulder.
Captures "turned around to talk to someone" or "checking something behind me."
 
**Session 7 — Face absent / out of frame (1 min)**
Stand up and walk away from the laptop.
Or lean so far to the side that your face exits the frame entirely.
Covers the "completely left the desk" violation case.
 
**Session 8 — Dim lighting + distracted (2 min)**
Repeat any of sessions 1–6 under dim lighting.
Lighting diversity in not_focused is just as important as in focused.
 
---
 
### Recording tips
 
- **Between pose changes, move naturally** — don't hard-cut. Transition frames
  (mid-turn, mid-reach) actually improve model robustness.
- **Don't have other people in frame** — model should learn your face only.
- **Aim for 600+ images per class before training.** Check counts after extraction.
- **If counts are uneven** (e.g. 700 focused, 400 not_focused) — record more clips
  for the smaller class before training. Class imbalance hurts F1 badly.
---