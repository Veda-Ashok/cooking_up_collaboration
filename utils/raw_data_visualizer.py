import json
import pandas as pd
import ast
import os
import pygame.image
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, Recipe
from overcooked_ai_py.mdp.actions import Action


if not os.path.exists('csv_visualized'):
    os.makedirs('csv_visualized')
    
Recipe.configure({"ingredients": ['onion', 'tomato']})

df = pd.read_csv('2019_hh_trials.csv')
visualizer = StateVisualizer()

for i in range(500):

    raw_state = json.loads(df.state[i])
    state_object = OvercookedState.from_dict(raw_state)

    grid_rep = ast.literal_eval(df.layout[i])
    formatted_grid = [row.replace('1', ' ').replace('2', ' ') for row in grid_rep]

    img = visualizer.render_state(state_object, grid = formatted_grid)

    pygame.image.save(img, f'csv_visualized/overcooked_frame_{i}.png')
