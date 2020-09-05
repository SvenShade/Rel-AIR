# Rel-AIR
Official repo for our paper, "A Closer Look at Generalisation in RAVEN", ECCV2020.

Code still needs to be refactored! This will happen shortly. If you want to play around with the code in the meantime, here are some steps to understanding it:

1. We downloaded the RAVEN-10k dataset (http://wellyzhang.github.io/project/raven.html#dataset), and processed it by reducing the size of each image to 80x80 (half-size). We do this instead of generating a new set at that size using the authors' code (there is an issue with that code which degrades image quality at small sizes). We also deleted information we weren't planning to use (e.g. data["meta_target"]).

1. Using the AIR module at https://pyro.ai/examples/air.html, we generated a second dataset, consisting of objects found by the module, and their position and scale latents.

1. We downloaded the original RAVEN code, available at https://github.com/WellyZhang/RAVEN, on which to build this project. Download this and replace the model folder with ours to run our code.

1. With those datasets (which we will also be uploading), one can train the models like so:

`python src/model/main.py --path <your_dataset_folder> --model <"ResNet", "Rel-Base", or "Rel-AIR"> --batch_size n --percent <% of training data to use> --trn_configs <one or more of cs, io, ud, lr, d4, d9, 4c> --tst_configs <one or more of cs, io, ud, lr, d4, d9, 4c> --multi_gpu <True or False> --epochs n --val_every n --test_every n`

Notes:
* If training Rel-AIR, it will expect the dataset to contain objects and latents. 
* You can train and test on any combination of scene configurations (e.g. train on Up-Down and test generalisation to Left-Right). If you want to train and test on the full dataset (all 7 configs), run main.py without the trn_configs and tst_configs arguments.
* Also see our supplementary material (in this repository) for full details on model architecture.
