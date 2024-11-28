# AAI3001 Final Project Team 6

### Contents
| **Files**                       | **Description**                                                                                          |
|---------------------------------|----------------------------------------------------------------------------------------------------------|
| `PreliminaryResearch.ipynb`          | Notebook that details our initial research into this topic                                          |
| `MainNotebook.ipynb`                 | Notebook for training our main model                                                                |
| `MainNotebook-Depth.ipynb`           | Notebook for training our secondary model based on depth information                                |
| `livepredict-single.py`              | Python file to get a live preview of the main model                                                 |
| `livepredict-dual.py`                | Live preview of both models (RGB and depth), with a heatmap overlay of the generated depth information |

| **Folders**                      | **Description**                                                                                          |
|---------------------------------|----------------------------------------------------------------------------------------------------------|
| `3dprint_depthpro_yolov11m`          | Contains model training and test results for the depth dataset                                       |
| `3dprint_yolov11m`                   | Contains model training and test results for the regular dataset                                     |
| `Checkpoints`                        | Folder for Apple's Depth Pro model. Download it from [link](https://sitsingaporetechedu-my.sharepoint.com/:u:/g/personal/2302822_sit_singaporetech_edu_sg/EarUQsqcFjhBle4mf87DVD4BuVpA4PcqrSdIr7X4MDN9hg?e=EZ8QTb) and extract it                             |
| `demo_videos`          | Contains backup demo videos, as well as raw footage to upload and test on UI                                      |
| `depth_pro`            | Contains required files to run [Apple's model](https://github.com/apple/ml-depth-pro/tree/main)                   |
| `utilities`            | Contains helper functions to split dataset, and other unused test functions               |

### Required Stuff
**Install these python libraries:**
   - `ultralytics`
   - `streamlit`
   - `opencv`
   - `flask`
   - *Optional: Please download the Apple Depth Pro mode from this [link](https://sitsingaporetechedu-my.sharepoint.com/:u:/g/personal/2302822_sit_singaporetech_edu_sg/EarUQsqcFjhBle4mf87DVD4BuVpA4PcqrSdIr7X4MDN9hg?e=EZ8QTb), and extract it to the checkpoints folder.*

### Running
- **Streamlit UI** `streamlit run ProjectUI.py`
- **Single Model Demo** `python livepredict-single.py`
- **Dual Model Demo** `python livepredict-dual.py`
- **Stream OpenCV Footage (Only works for ChenYi)** `python livepredict-single-flask.py`

# Introduction and Problem Statement
TODO

# Research Questions and Project Objectives
TODO

# Methodology and Proposed Solution
TODO
### Dataset

### Model Development

### Model Deployment

### User Interface
PUT SCREENSHOTS HERE

# Experiments and Results Analysis
TODO
