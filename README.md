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
| `app`                  | Contains streamlit UI files|

### Required Stuff
**Install these python libraries:**
   - `ultralytics`
   - `streamlit`
   - `opencv`
   - `flask`
   - *Optional: Please download the Apple Depth Pro mode from this [link](https://sitsingaporetechedu-my.sharepoint.com/:u:/g/personal/2302822_sit_singaporetech_edu_sg/EarUQsqcFjhBle4mf87DVD4BuVpA4PcqrSdIr7X4MDN9hg?e=EZ8QTb), and extract it to the checkpoints folder.*

### Running
- **Streamlit UI** `streamlit run /app/home.py`
- **Single Model Demo** `python livepredict-single.py`
- **Dual Model Demo** `python livepredict-dual.py`
- **Stream OpenCV Footage (Only works for ChenYi as camera is local)** `python livepredict-single-flask.py`

# Introduction and Problem Statement
3D prints might take a few minutes, or several days to complete. The idea of using a camera to remotely monitor prints stems from this. Nearly all 3D printers on the market now have built-in cameras for the remote monitoring of 3D prints. It offers users a piece of mind through the ability to monitor their from wherever they are.

<img src="https://i.ytimg.com/vi/zbCap3a_tFA/maxresdefault.jpg" width="600px">

**An example of a failure. The left image shows a 'spaghetti' failure, and the right image is a clogged nozzle causing a ['blob of death'](https://docs.vorondesign.com/community/troubleshooting/120decibell/blob_of_death.html)**

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
We will detail how we went about obtaining our dataset

#### Initial Dataset (350 Images)
Our initial dataset consists of 350 Images. These are recorded from the 3D printer's built in webcam. We obtain our images as snapshots from a mix of timelapse and regular videos. These have been roughly split into 65:25:10 (train/val/test), of these, 157 have at least one failure in them, and the remaining are background images.

- #### Timelapse Videos
These captures movement across the entire print cycle, and are representative of an entire print, which takes more than an hour on averages. Frames are captured at the start of each layer, where each layer is 0.2mm in height. Therefore an object of 1cm will have 5 frames captures. However, a shortfall of this method is key moments, such as the start of a print failure occuring might get missed if they do not occur at the start of the new layer.

- #### Regular Videos
These captures movement across a specific moment in the print cycle. We record the entire footage and pick out key moments. We use this to capture the initial moments where the print failure occurs. As this is a regular 30FPS video, we are able to capture more frames and populate our dataset with these key moments. As 30FPS will generate too many similar images, we record them in a 15:1 ratio (We capture one frame for every 15 frame of video). This is till much faster than timelapse, which might only capture a single image every 2+ mins.

Unfortunately, we were unable to organically capture an actual failure. Hence we have to simulate one. In particular, we will focus on ["spaghetti"](https://help.prusa3d.com/article/spaghetti-monster_1999) failures, which are the most common form of failure that has no recovery. I.e. encountering one will result in the print having to be completely discarded and restarted.

The failures from this initial dataset is simulated by printing a very thin but tall object. When it inevitably falls over, filament will get dragged everywhere.

**Here is an sample of our annotated initial dataset**

![Initial Dataset](https://i.ibb.co/VmKsFjV/Screenshot-2024-11-28-132652.jpg)

While our model works, and was able to reliably detect similar forms of failure, we felt that we did not bring anything new to this area of application. As such, we decided to shift our focus towards **Print Bed Foriegn Object Detection**. Here, we attempt to develop a model to reliably detect objects on the print surface before a print starts. This is important for several reasons:

**1. Current 3D Printers do not have a way to check if the print bed is clear before starting**
   - The user might have forgotten to remove a print after is completed
   - There might be leftover "purged filament" that fell onto the printbed

**2. If a prints start when there is an object on the bed, it will nearly always end in print and mechanical failure.**
### New Model Development

#### Updated Dataset (820 Images)
Based on the initial testing, we have updated the dataset with 280 more images. These are obtained similarly to the initial dataset, but using seperate camera for another slightly different perspective. These additional images are split into (80:10:10) where xx of xx images have at least one failure in them. The failures simulated are also done using two different methods:

We designed [two files](https://cad.onshape.com/documents/f5fa610d41301785390c590c/w/85f44a550e1a6889dc6a74d6/e/ebb2803dda6e79d8e5a7e7c0?renderMode=0&uiState=6743750fd0d06252d2c72f97) that when printed will result in failure:

- File-1: Unsupported cantiliver. This simulates a spagehetti failure mid-print. This can occur of the print belts skip, or due to a bad design which causes the printer to print "mid-air".

- File-2: Lack of adhesion. This simulates when the print surface is dirty, and the first layer does not stick to it, similar to the initial dataset. The filament will be dragged around a large area.

When researching for methods to improve detection rates, we found Apple's Depth Pro, which is a zero-shot metric monocular depth estimation based on vision transformers. When testing this model, we noticed that it was able to very clearly identify tall objects with ease. As we have discovered with our previous model, our dataset is still extremly limted, and it cannot generalize errors well, such as those of different coloured fillament. By passing our dataset through this model, we can obtain the estimated depth from the camera, and feed the images through our YOLO model.

![Depth Dataset](https://i.ibb.co/D8wXVKc/val-batch0-labels.jpg)

In actual testing however, it performed very poorly, especially when trying to identify purged filament on the build plate. This is because their relative height is very small, making it not stand out. Based on this, we decided the best way forward is to use both a regular YOLO model trained on RGB images, and pass live video feed through Apple's depth pro model, then feed it to a depth trained YOLO model.

**Here is an example of the combined model in action:**

![GIF](https://github.com/ExpiredTapWater/AAI3001-Final-Project/blob/main/demo_videos/side-by-side-gif.gif?raw=true)


### Model Deployment
We host a streamlit application on one of our member's desktop, with support for GPU acceleration. To replicate a full web hosted service without actually exposing any ports to the internet, we use a combination of a VPN service, local DNS server, reverse proxy and a free domain ("local.diskstation.me") with an SSL cert. Thus once connected, any user will be able to directly access the serice at "aai3001.local.diskstation.me". Users can upload their images and it will return the model's prediction. If a failure is detected, a notification will be sent to a user's phone via the 'ntfy' service. 

**Below is a screenshot of the interface:**

![UI](https://i.ibb.co/kHTqPf4/Screenshot-2024-11-30-124520.jpg)

Our original objective would be to provide a platform where users could either stream their webcam footage, or provide a IP address to a camera stream. This is because most camera equiped 3D printers stream via IP, such as via the RTSP protocol. However, we were unable to get this feature working in time for submission, thus we had to run another seperate flask app to obtain the annotated feed from opencv using `python livepredict-single-flask.py`. 

**Below is a screenshot of this funtion:**

![Raw UI](https://i.ibb.co/GCwd3mp/Screenshot-2024-11-30-124736.jpg)


**This raw stream can then be viewed from any compatible player, for example, from HomeAssistant, demonstrated here:**

![HA UI](https://i.ibb.co/QkdbRgq/Screenshot-2024-11-30-125043.jpg)

Adding an automatic pause function will be trivial, such as by checking a seperate webpage for the output, and sending a stop command via an [MQTT wrapper library](https://github.com/greghesp/ha-bambulab) for this line of 3D printers.

# Experiments and Results Analysis
TODO
