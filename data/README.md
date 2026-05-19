# Data directory

Do not commit the xBD / xView2 dataset to this repository.

Expected local structure:

```text
data/
└── raw/
    └── train/
        ├── images/
        ├── labels/
        └── targets/
```

The Challenge training set contains paired files:

```text
<disaster>_<id>_pre_disaster.png
<disaster>_<id>_post_disaster.png
<disaster>_<id>_pre_disaster.json
<disaster>_<id>_post_disaster.json
<disaster>_<id>_pre_disaster_target.png
<disaster>_<id>_post_disaster_target.png
```

The dataset must be downloaded from the official xView2 source. Do not redistribute data files or generated large archives.
