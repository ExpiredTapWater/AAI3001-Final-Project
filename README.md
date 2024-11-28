# 3D Printing Failure Detection
### AAI3001 Final Project Team 6

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
3D prints might take a few minutes, or several days to complete. The idea of using a camera to remotely monitor prints stems from this. Nearly all 3D printers on the market now have built-in cameras for the remote monitoring of 3D prints. It offers users a piece of mind through the ability to monitor their from wherever they are.

<img src="https://i.ytimg.com/vi/zbCap3a_tFA/maxresdefault.jpg" width="600px">

**An example of a failure. The left image shows a 'spaghetti' failure, and the right image is a clogged nozzle causing a['blob of death'](https://docs.vorondesign.com/community/troubleshooting/120decibell/blob_of_death.html)**

![Failure](https://i.ibb.co/bBnCPhT/Screenshot-2024-11-28-130358.jpg)

With the growth in AI in recent years, and the exceeding number of printers already equipped with a camera, manufacturers have started to include a form of object detection to identify potential issues with prints, and stop them if needed. As this is a relatively new application of AI in this field, there are still many opportunities to apply our knowledge to improve this.

### Identified Problems
Current implementations face issues such as:
- Proprietary software locked to specific hardware
- Trained using generic data resulting in overly sensitive/insensitive detection
- Limited User control

# Research Questions and Project Objectives
When researching models, we found that the YOLO model is able to make use of unlabled images a 'background' images for training. This is a unique opportunity to explore the idea of applying YOLO for an object recognition task will nearly always have a static background. 

**To illustrate, here is the view from the built in camera from a Bambu Lab 3D Printer:**
![Chamber Camera](https://preview.redd.it/an-easy-way-to-get-your-x1c-camera-available-elsewhere-on-v0-w8j9iep9oeyd1.png?width=556&format=png&auto=webp&s=f7ce711d45db3a4fc4ec177bca1a46137d146643)

As the camera is located inside the printer, except for different lighting conditions, the background will never change. In further research, companies offering print monitoring software have trained their model using generic images available online where the background is not constant. 

**An example of such an image is this:**

![Failure 2](https://i.ibb.co/ZhcrzDT/Screenshot-2024-11-28-131744.jpg)

Using this knowledge, we will attempt to train a simple model using YOLO to detect print failures, and determine if there are further opportunities to explore

# Methodology and Proposed Solution
TODO
### Dataset

### Model Development

### Model Deployment

### User Interface
PUT SCREENSHOTS HERE

# Experiments and Results Analysis
TODO
