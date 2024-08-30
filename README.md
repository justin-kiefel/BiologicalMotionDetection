# Motion Classification: Biological vs. Non-Biological

This project explores the classification of motion in videos as either biological (e.g., a cat walking) or non-biological (e.g., a plastic bag in the wind). The goal is to compare the effectiveness of two approaches: one that leverages kinematic features and another that uses sequences of poses constructed from distances between keypoints detected from sparse optical flow in each frame of the video.

## Motivation

Research has shown that both humans and animals have an innate ability to perceive biological motion differently from other types of motion. This ability is crucial for survival, as it helps in recognizing living beings, which may indicate potential threats or social interactions. Understanding and replicating this ability in machines has significant implications in fields like search and rescue, where identifying living beings quickly can be life-saving. Similarly, in the service industry, robots equipped with this capability can better interact with humans and animals, enhancing their efficiency and safety.

## Project Overview

### Approach

1. **Kinematic Features:**
   - **Description:** Kinematic features such as velocity, acceleration, and trajectory curvature are extracted from the video frames. These features capture the dynamics of the motion, providing a rich set of data for classification.
   - **Model:** We train Support Vector Machines (SVMs) and other traditional classifiers on these kinematic features to distinguish between biological and non-biological motion.

2. **Sequences of Poses:**
   - **Description:** Sparse optical flow is used to detect keypoints in each frame of the video. The distances between these keypoints are then computed to form a sequence of poses over time, capturing the motion structure.
   - **Model:** An LSTM (Long Short-Term Memory) network is trained on the sequence of poses. The LSTM's ability to capture temporal dependencies makes it well-suited for this task.

### Data Processing

- **Keypoint Detection:** Keypoints are detected using sparse optical flow, which tracks the motion of specific points between frames.
- **Feature Extraction:** For kinematic analysis, features such as velocity, acceleration, and curvature are extracted. For pose-based analysis, the distances between detected keypoints form the basis of the feature set.
- **Sequence Construction:** The sequence of poses is constructed by calculating the pairwise distances between keypoints for each frame and stacking these into a time series.

### Evaluation

The project evaluates the performance of the LSTM model trained on pose sequences against traditional classifiers (e.g., SVMs) trained on kinematic features. Metrics such as accuracy, precision, recall, and F1-score are used to compare the models.

## Results

The results section will discuss the performance of the models, highlighting the strengths and weaknesses of each approach. Preliminary findings indicate that the LSTM model, with its ability to capture temporal patterns in motion, may outperform traditional kinematic-based models in distinguishing biological from non-biological motion.
