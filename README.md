# DIEM
Dimension Insensitive Euclidean Metric

This repository contains the Matlab functions to correctly compute the Dimension Insensitive Euclidean Metric.

For details about the metric, please refer to the following paper: https://arxiv.org/abs/2407.08623

The repository contains 3 files:
- DIEM_Stat.m --> This function computes the statistical parameters of the DIEM distribution given a dimension N and variables range (minV,maxV).
- getDIEM.m --> This function computes the DIEM between any pairs of vectors or matrices.
- Example_DIEM.m --> This script provides a simple example on how to run and use the DIEM_Stat.m and getDIEM.m functions.
- getCosineSimilarity.m --> This function computes the Cosine Similarity between any pairs of vectors or matrices.
- randu_sphere.m --> This function allows the generation of unformly distributed random points on a shpere.
- DIEM_paper_results.m --> Run this script to reproduce the results of the the paper on arXiv.
  - In order to run this code you should also download two additional .csv files containing the vector embeddings for the LLM case-study. You can download such files from this link: https://drive.google.com/drive/folders/1LvvYO7YfgsR0jJ3Je3wIkUsl1sdKRcoe?usp=sharing
  - Place the 2 .csv files in the TextEmbeddings folder.
 

For any question or issue running the code, please feel free to reach out to me at ftessari@mit.edu
