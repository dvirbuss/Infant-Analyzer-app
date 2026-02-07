# Infant Motor Assessment Project

This repository contains the full implementation, documentation, and output for an engineering capstone project focused on automated infant motor development assessment using side-view videos and pose estimation.

## 📁 Folder Structure

- **articles/**
  - A collection of academic articles and papers that inspired and supported the literature review.

- **code/**
  - User Guide(Word file)
  - All Python source code used in the system.
  - The main GUI application is in `PE_gui.py`, which orchestrates the video input, pose estimation, joint analysis, and visualization pipeline.

- **documents/**
  - Final deliverables submitted for the project:
    - Project report
    - Poster
    - Project presentation (slides)
    - Additional supporting documents

- **infant data/**
  - Contains labeled or preprocessed video datasets and keypoint files used during testing and training.

- **models/**
  - Trained YOLOv8 pose estimation models for infant keypoints.
  - Models are named by training date and stored for reproducibility.

- **output results/**
  - Example output of a complete run on one video input, including:
    - Annotated video
    - TSV files (keypoints, angles)
    - Milestone reports
    - Visual graphs
