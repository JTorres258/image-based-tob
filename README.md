# **Advancing Newborn Care: Precise Birth Time Detection Using AI-Driven Thermal Imaging with Adaptive Normalization**

This is the official Github repository for the paper "Advancing Newborn Care: Precise Birth Time Detection Using AI-Driven Thermal Imaging with Adaptive Normalization".

## Abstract

Around 5-10% of newborns need assistance to start breathing. Currently, there is a lack of evidence-based research, objective data collection, and opportunities for learning from real newborn resuscitation emergency events. Generating and evaluating automated newborn resuscitation algorithm activity timelines relative to the Time of Birth (ToB) offers a promising opportunity to enhance newborn care practices. Given the importance of prompt resuscitation interventions within the "golden minute" after birth, having an accurate ToB with second precision is essential for effective subsequent analysis of newborn resuscitation episodes. Instead, ToB is generally registered manually, often with minute precision, making the process inefficient and susceptible to error and imprecision. In this work, we explore the fusion of Artificial Intelligence (AI) and thermal imaging to develop the first AI-driven ToB detector. The use of temperature information offers a promising alternative to detect the newborn while respecting the privacy of healthcare providers and mothers. However, the frequent inconsistencies in thermal measurements, especially in a multi-camera setup, make normalization strategies critical. Our methodology involves a three-step process: first, we propose an adaptive normalization method based on Gaussian mixture models (GMM) to mitigate issues related to temperature variations; second, we implement and deploy an AI model to detect the presence of the newborn within the thermal video frames; and third, we evaluate and post-process the model's predictions to estimate the ToB. A precision of 91.8% and a recall of 92.8% are reported in the detection of the newborn within thermal frames during performance evaluation. Our approach achieves an absolute median deviation of 3.5 seconds in estimating the ToB relative to the manual annotations.

## Simulated Data

Due to privacy and data treatment policies, the sensitive dataset employed in this paper cannot be shared. However, to compensate for this setback, we provide the inference pipeline along with simulated data. Simulation sessions took place at Stavanger University Hospital. Five people were considered in this process, playing the role of the mother, her partner, the midwife, a midwife assistant, and an extra midwife. The following equipement was used:

- A pregnancy simulation belly.
- A manikin filled with warm water (around 40°C).
- A infrared thermometer with laser pointer, to ensure the in-use manikin temperature is high enough.

If required, more simulated examples can be provided.

## Content

The content of this repository is organized as follows:
- [Code](./code/): Main code to execute the inference.
    - [pipeline.py](./code/pipeline.py) is the main script of this repository. It takes all the frames from a 50-second thermal clip of a simulated birth and estimates the Time of Birth. As output, it generates a `.TXT` file with the raw scores, a `.PNG` image with a visualization of the results, and a `.MP4` video for interpretability and explanaibility of the model's making-decision process.
    - [Model](./code/model/) contains the label mapping.
    - [Utils](./code/utils/) contains useful function required during the inference, such as the data loader, the estimation of the Gaussian Mixture Model, the timeline creation, and the GradCAM algorithm.
    - [Output](./code/output/) contains the files generated when running the _pipeline.py_ script.

- [Data](./data/): Dataset employed for running inference.
    - [raw_images](./data/raw_images) must contains the raw files extracted from the thermal camera. Raw data is represented as 14-bit grayscale images. This folder will be generated when downloading and extracting the data.
    - [anns.txt](./data/anns.txt) includes the manual annotations of the thermal video.
    - [colormap_video.mp4](./data/colormap_video.mp4) is a colorful visualization of the data employed in this repository. The temperature values have been clipped within the range [20,40] Celsius degrees.

## Downloads

The simulated data can be downloaded here:
- [Data](https://drive.google.com/file/d/1IhgQkdQyCte_vEX0C1SAddFNbhT8zERB/view?usp=sharing)

Make sure you extract the files in the corresponding `raw_images` folder. This step is required to keep the relative paths of the code consistent.

## Citation

If you use this code in your research or project, please cite the following GitHub repository:

```bibtex
@misc{garcia-torres2024,
  author = {García-Torres, Jorge},
  title = {Advancing Newborn Care: Precise Birth Time Detection Using AI-Driven Thermal Imaging with Adaptive Normalization},
  year = {2024},
  howpublished = {\url{https://github.com/JTorres258/image-based-tob}}, 
}
```

Please include this citation in any publication or presentation that uses this code.

Note: Once the paper is accepted and published, please update your citations to reference the final published version.
