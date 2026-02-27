"Usage: From root directory, run python3 utils/raw_data_visualizer.py <input_file.csv>"

import json, ast, os, glob, sys
import pandas as pd
import cv2 as cv
import pygame.image
from pathlib import Path
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, Recipe
from overcooked_ai_py.mdp.actions import Action


if len(sys.argv) < 2:
    print("Please enter an input file path.")
    print("Usage: python3 raw_data_visualizer.py <input_file.csv>")
    sys.exit(1)

INPUT_FILE = sys.argv[1]
# INPUT_FILE = '2019_hh_trials.csv'
STEM = Path(INPUT_FILE).stem

video_count = len(list(Path('csv_visualized').glob(f'{STEM}_*'))) if Path('csv_visualized').exists() else 0
img_dir = Path(f'csv_visualized/{STEM}_{video_count}')
vid_dir = Path(f'gameplays/{STEM}_{video_count}.mp4')

img_dir.mkdir(parents = True, exist_ok = True)
vid_dir.parent.mkdir(parents = True, exist_ok = True)
    
Recipe.configure({"ingredients": ['onion', 'tomato']})
df = pd.read_csv(f'./data/{INPUT_FILE}')
visualizer = StateVisualizer()

for i in range(500):
    raw_state = json.loads(df.state[i])
    state_object = OvercookedState.from_dict(raw_state)
    grid_rep = ast.literal_eval(df.layout[i])
    #formatted_grid = [row.replace('1', ' ').replace('2', ' ') for row in grid_rep] # Need this for parsing the csv data

    img = visualizer.render_state(state_object, grid = grid_rep) # if using hh trial csv, change grid_rep to formatted_grid
    pygame.image.save(img, f'{img_dir}/overcooked_frame_{i:04d}.png')

# make the video
img_array = []
for f in sorted(glob.glob(f'{img_dir}/overcooked_frame_*.png')):
    img = cv.imread(f)
    img_array.append(img)

if img_array:
    h, w, d = img_array[0].shape
    size = (w, h)
    vid = cv.VideoWriter(vid_dir, cv.VideoWriter_fourcc(*'mp4v'), 10, size)
   
    for i in range(len(img_array)):
        # for j in range(3): # for smoother rendering, if needed
        vid.write(img_array[i])
vid.release()

