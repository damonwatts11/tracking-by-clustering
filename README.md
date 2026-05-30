# Tracking-by-Clustering

TU Dresden — Machine Learning for Computer Vision — Team Project

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=tbc --display-name "Python (tbc)"
```

## Running

Open `main.ipynb` in VS Code or Jupyter, select kernel **Python (tbc)**, run all cells.

## Structure

```
tbc/
  synthesis.py   # SimConfig, SyntheticDataset, generate/save/load
  geometry.py    # wrap, torus distance
  motion.py      # simulate, Trajectory
  collision.py   # elastic collision resolution
  observation.py # noisy point cloud sampling
  viz.py         # Plotly spacetime + animation
main.ipynb       # demonstration notebook
```
