# Extended Phystwin Pipeline With New Stuff

Please see original phystwin [repo](https://github.com/Jianghanxiao/PhysTwin) for the original pipeline. This repo contains an extended version of the original pipeline with the following new features
1. Support pose estimation with Dust3R or Depth Anything 3
2. Support depth estimation (does not need RGBD input, can work with RGB input only) with Dust3R or Depth Anything 3
3. Added a new phase for depth refinement with a new "temporal depth smoothing" model, that can improve depth estimation results by leveraging temporal information across frames. Repo contains both data preparation, training, and inference code for this new model.
4. Added new phases for alignment refinement, which normalizes extrinsics/intrinsics and the ground plane across frames, and can further improve the quality of the final results.
5. Added random misc changes to the overall pipeline
  - Allow support for different input resolutions between RGB and depth
  - Some consistency improvements for using `data_config.csv` to specify input data
  - Performance improvements so the pipeline doesnt crash on worse GPUs.
  - Other minor code fixes that I ran into while testing the pipeline on new datasets
6. Added random misc scripts for visualization, evaluation, and data processing


Will add more details to this README later...