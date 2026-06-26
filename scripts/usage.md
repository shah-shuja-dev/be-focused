## Pre requisites

Create the required directories and structure

```
    mkdir data\raw
    mkdir data\processed\focused
    mkdir data\processed\not_focused
```

- use `extrct_frames.py` to extract and process the frames into 600x600 images and save them in the data/processed folder.
  Usage script :

```
   PWRSHELL: # Process all focused clips
   Get-ChildItem data\raw\focused_*.avi | ForEach-Object {
       python scripts\extract_frames.py --video $_.FullName --label focused
   }

   # Process all not_focused clips
   Get-ChildItem data\raw\not_focused_*.avi | ForEach-Object {
       python scripts\extract_frames.py --video $_.FullName --label not_focused
   }
```

- use `capture_dataset.py` to record new clips and save them in the data/raw folder.

```
    # Record one focused clip (2 min) — run this 5-6 times with different poses each time
    python scripts\capture_dataset.py --label focused --duration 120

    # Record one not_focused clip — run this 7-8 times
    python scripts\capture_dataset.py --label not_focused --duration 120
```
