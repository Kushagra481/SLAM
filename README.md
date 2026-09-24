https://www.kaggle.com/code/kushagra678/basic-gsplat-start
                 TUM RGB-D
                     │
                     ▼
               RGB + Depth
                     │
                     ▼
             Gaussian Map
                     │
                     ▼
              Frame t
                     │
                     ▼
             Pose Tracking
                     │
                     ▼
             Estimated Pose
                     │
                     ├──────────────┐
                     │              │
                     ▼              ▼
             RGB-D Rendering    GT Pose
                     │              │
                     ▼              │
                 Loss               │
                     │              │
                     ▼              ▼
              Update Gaussians   Evaluation
                     │
                     ▼
                  Frame t+1
