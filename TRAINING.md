You can train your own model by following these instructions:

## Step 1: Install COCO API
To train new models you will need to install the [COCO Python API](https://github.com/cocodataset/cocoapi). Unfortunately installing this package via pip often leads to build errors, but you can install it from source like this:

```bash
cd ~
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
python setup.py install
```

## Step 2: Preparing the data

### Visual Genome
Run the following script to download and unpack the relevant parts of the Visual Genome dataset:

```bash
bash scripts/download_vg.sh
```

This will create the directory `datasets/vg` and will download about 15 GB of data to this directory; after unpacking it will take about 30 GB of disk space.

After downloading the Visual Genome dataset, we need to preprocess it. This will split the data into train / val / test splits, consolidate all scene graphs into HDF5 files, and apply several heuristics to clean the data. In particular we ignore images that are too small, and only consider object and attribute categories that appear some number of times in the training set; we also igmore objects that are too small, and set minimum and maximum values on the number of objects and relationships that appear per image.

```bash
python scripts/preprocess_vg.py
```

This will create files `train.h5`, `val.h5`, `test.h5`, and `vocab.json` in the directory `datasets/vg`.

### COCO
Run the following script to download and unpack the relevant parts of the COCO dataset:

```bash
bash scripts/download_coco.sh
```

This will create the directory `datasets/coco` and will download about 21 GB of data to this directory; after unpacking it will take about 60 GB of disk space.

## Step 3: Train a model

Now you can train a new model by running the script:

```bash
python scripts/train.py
```

By default this will train a model on COCO, periodically saving checkpoint files `checkpoint_with_model.pt` and `checkpoint_no_model.pt` to the current working directory. The training script has a number of command-line flags that you can use to configure the model architecture, hyperparameters, and input / output settings:

### Optimization

- `--batch_size`: How many pairs of (scene graph, image) to use in each minibatch during training. Default is 32.
- `--num_iterations`: Number of training iterations. Default is 1,000,000.
- `--learning_rate`: Learning rate to use in Adam optimizer for the generator and discriminators; default is 1e-4.
- `--eval_mode_after`: The generator is trained in "train" mode for this many iterations, after which training continues in "eval" mode. We found that if the model is trained exclusively in "train" mode then generated images can have severe artifacts if test batches have a different size or composition than those used during training.

### Dataset options

- `--dataset`: The dataset to use for training; must be either `coco` or `vg`. Default is `coco`.
- `--image_size`: The size of images to generate, as a tuple of integers. Default is `64,64`. This is also the resolution at which scene layouts are predicted.
- `--num_train_samples`: The number of images from the training set to use. Default is None, which means the entire training set will be used.
- `--num_val_samples`: The number of images from the validation set to use. Default is 1024. This is particularly important for the COCO dataset, since we partition the COCO validation images into our own validation and test sets; this flag thus controls the number of COCO validation images which we will use as our own validation set, and the remaining images will serve as our test set.
- `--shuffle_val`: Whether to shuffle the samples from the validation set. Default is True.
- `--loader_num_workers`: The number of background threads to use for data loading. Default is 4.
- `--include_relationships`: Whether to include relationships in the scene graphs; default is 1 which means use relationships, 0 means omit them. This is used to train the "no relationships" ablated model.

**Visual Genome options**:
These flags only take effect if `--dataset` is set to `vg`:

- `--vg_image_dir`: Directory from which to load Visual Genome images. Default is `datasets/vg/images`.
- `--train_h5`: Path to HDF5 file containing data for the training split, created by `scripts/preprocess_vg.py`. Default is `datasets/vg/train.h5`.
- `--val_h5`: Path to HDF5 file containing data for the validation split, created by `scripts/preprocess_vg.py`. Default is `datasets/vg/val.h5`.
- `--vocab_json`: Path to JSON file containing Visual Genome vocabulary, created by `scripts/preprocess_vg.py`. Default is `datasets/vg/vocab.json`.
- `--max_objects_per_image`: The maximum number of objects to use per scene graph during training; default is 10. Note that `scripts/preprocess_vg.py` also defines a maximum number of objects per image, but the two settings are different. The max value in the preprocessing script causes images with more than the max number of objects to be skipped entirely; in contrast during training if we encounter images with more than the max number of objects then they are randomly subsampled to the max value as a form of data augmentation.
- `--vg_use_orphaned_objects`: Whether to include objects which do not participate in any relationships; 1 means use them (default), 0 means skip them.

**COCO options**:
These flags only take effect if `--dataset` is set to `coco`:

- `--coco_train_image_dir`: Directory from which to load COCO training images; default is `datasets/coco/images/train2017`.
- `--coco_val_image_dir`: Directory from which to load COCO validation images; default is `datasets/coco/images/val2017`.
- `--coco_train_instances_json`: Path to JSON file containing object annotations for the COCO training images; default is `datasets/coco/annotations/instances_train2017.json`.
- `--coco_train_stuff_json`: Path to JSON file containing stuff annotations for the COCO training images; default is `datasets/coco/annotations/stuff_train2017.json`.
- `--coco_val_instances_json`: Path to JSON file containing object annotations for COCO validation images; default is `datasets/coco/instances_val2017.json`.
- `--coco_train_instances_json`: Path to JSON file containing stuff annotations for COCO validation images; default is `datasets/coco/stuff_val2017.json`.
- `--instance_whitelist`: The default behavior is to train the model to generate all object categories; however by passing a comma-separated list to this flag we can train the model to generate only a subset of object categories.
- `--stuff_whitelist`: The default behavior is to train the model to generate all stuff categories (except other, see below); however by passing a comma-separated list to this flag we can train the model to generate only a subset of stuff categories.
- `--coco_include_other`: The COCO-Stuff annotations include an "other" category for objects which do not fall into any of the other object categories. Due to the variety in this category I found that the model was unable to learn it, so setting this flag to 0 (default) causes COCO-Stuff annotations with the "other" category to be ignored. Setting it to 1 will include these "other" annotations.
- `--coco_stuff_only`: The 2017 COCO training split contains 115K images. Object annotations are provided for all of these images, but Stuff annotations are only provided for 40K of these images. Setting this flag to 1 (default) will only train using images for which Stuff annotations are available; setting this flag to 0 will use all 115K images for training, including Stuff annotations only for the images for which they are available.

### Generator options
These flags control the architecture and loss hyperparameters for the generator, which inputs scene graphs and outputs images.

- `--mask_size`: Integer giving the resolution at which instance segmentation masks are predicted for objects. Default is 16. Setting this to 0 causes the model to omit the mask prediction subnetwork, instead using the entire object bounding box as the mask. Since mask prediction is differentiable the model can predict masks even when the training dataset does not provide masks; in particular Visual Genome does not provide masks, but all VG models were trained with `--mask_size 16`.
- `--embedding_dim`: Integer giving the dimension for the embedding layer for objects and relationships prior to the first graph convolution layer. Default is 128.
- `--gconv_dim`: Integer giving the dimension for the vectors in graph convolution layers. Default is 128.
- `--gconv_hidden_dim`: Integer giving the dimension for the hidden dimension inside each graph convolution layer; this is the dimension of the candidate vectors V^s_i and V^s_o from Equations 1 and 2 in the paper. Default is 512.
- `--gconv_num_layers`: The number of graph convolution layers to use. Default is 5.
- `--mlp_normalization`: The type of normalization (if any) to use for linear layers in MLPs inside graph convolution layers and the box prediction subnetwork. Choices are  'none' (default), which means to use no normalization, or 'batch' which means to use batch normalization.
- `--refinement_network_dims`: Comma-separated list of integers specifying the architecture of the cascaded refinement network used to generate images; default is `1024,512,256,128,64` which means to use five refinement modules, the first with 1024 feature maps, the second with 512 feature maps, etc. Spatial resolution of the feature maps doubles between each successive refinement modules.
- `--normalization`: The type of normalization layer to use in the cascaded refinement network. Options are 'batch' (default) for batch normalization, 'instance' for instance normalization, or 'none' for no normalization.
- `--activation`: Activation function to use in the cascaded refinement network; default is `leakyrelu-0.2` which is a Leaky ReLU with a negative slope of 0.2. Can also be `relu`.
- `--layout_noise_dim`: The number of channels of random noise that is concatenated with the scene layout before feeding to the cascaded refinement network. Default is 32.

**Generator Losses**: These flags control the non-adversarial losses used to to train the generator:

- `--l1_pixel_loss_weight`: Float giving the weight to give L1 difference between generated and ground-truth image. Default is 1.0.
- `--mask_loss_weight`: Float giving the weight to give mask prediction in the overall model loss. Setting this to 0 (default) means that masks are weakly supervised, which is required when training on Visual Genome. For COCO I found that setting `--mask_loss_weight` to 0.1 works well.
- `--bbox_pred_loss_weight`: Float giving the weight to assign to regressing the bounding boxes for objects. Default is 10.

### Discriminator options
The generator is trained adversarially against two discriminators: an patch-based image discriminator ensuring that patches of the generated image look realistic, and an object discriminator that ensures that generated objects are realistic. These flags apply to both discriminators:

- `--discriminator_loss_weight`: The weight to assign to discriminator losses when training the generator. Default is 0.01.
- `--gan_loss_type`: The GAN loss function to use. Default is 'gan' which is the original GAN loss function; can also be 'lsgan' for least-squares GAN or 'wgan' for Wasserstein GAN loss.
- `--d_clip`: Value at which to clip discriminator weights, for WGAN. Default is no clipping.
- `--d_normalization`: The type of normalization to use in discriminators. Default is 'batch' for batch normalization, but like CRN normalization this can also be 'none' or 'instance'.
- `--d_padding`: The type of padding to use for convolutions in the discriminators, either 'valid' (default) or 'same'.
- `--d_activation`: Activation function to use in the discriminators. Like CRN the default is `leakyrelu-0.2`.

**Object Discriminator**: These flags only apply to the object discriminator:

- `--d_obj_arch`: String giving the architecture of the object discriminator; the semantics for architecture strings [is described here](https://github.com/jcjohnson/sg2im-release/blob/master/sg2im/layers.py#L116).
- `--crop_size`: The object discriminator crops out each object in images; this gives the spatial size to which these crops are (differentiably) resized. Default is 32.
- `--d_obj_weight`: Weight for real / fake classification in the object discriminator. During training the weight given to fooling the object discriminator is `--discriminator_loss_weight * --d_obj_weight`. Default is 1.0
- `--ac_loss_weight`: Weight for the auxiliary classifier in the object discriminator that attempts to predict the object category of objects; the weight assigned to this loss is `--discriminator_loss_weight * --ac_loss_weight`. Default is 0.1.

**Image Discriminator**: These flags only apply to the image discriminator:

- `--d_img_arch`: String giving the architecture of the image discriminator; the semantics for architecture strings [is described here](https://github.com/jcjohnson/sg2im-release/blob/master/sg2im/layers.py#L116).
- `--d_img_weight`: The weight assigned to fooling the image discriminator is `--discriminator_loss_weight * --d_img_weight`. Default is 1.0.

### Output Options
These flags control outputs from the training script:

- `--print_every`: Training losses are printed and recorded every `--print_every` iterations. Default is 10.
- `--timing`: If this flag is set to 1 then measure and print the time that each model component takes to execute.
- `--checkpoint_every`: Checkpoints are saved to disk every `--checkpoint_every` iterations. Default is 10000. Each checkpoint contains a history of training losses, a history of images generated from training-set and val-set scene graphs,  the current state of the generator, discriminators, and optimizers, as well as all other state information needed to resume training in case it is interrupted. We actually save two checkpoints: one with all information, and one without model parameters; the latter is much smaller, and is convenient for exploring the results of a large hyperparameter sweep without actually loading model parameters.
- `--output_dir`: Directory to which checkpoints will be saved. Default is current directory.
- `--checkpoint_name`: Base filename for saved checkpoints; default is 'checkpoint', so the filename for the checkpoint with model parameters will be 'checkpoint_with_model.pt' and the filename for the checkpoint without model parameters will be 'checkpoint_no_model.pt'.
- `--restore_from_checkpoint`: Default behavior is to start training from scratch, and overwrite the output checkpoint path if it already exists. If this flag is set to 1 then instead resume training from the output checkpoint file if it already exists. This is useful when running in an environment where jobs can be preempted.
- `--checkpoint_start_from`: Default behavior is to start training from scratch; if this flag is given then instead resume training from the specified checkpoint. This takes precedence over `--restore_from_checkpoint` if both are given.

## (Optional) Step 4: Strip Checkpoint
Checkpoints saved by train.py contain not only model parameters but also optimizer states, losses, a history of generated images, and other statistics. This information is very useful for development and debugging models, but makes the saved checkpoints very large. You can use the script `scripts/strip_checkpoint.py` to remove all this extra information from a saved checkpoint, and save only the trained models:

```bash
python scripts/strip_checkpoint.py \
  --input_checkpoint checkpoint_with_model.pt \
  --output_checkpoint checkpoint_stripped.pt
```

## Step 5: Run the model
You can use the script `scripts/run_model.py` to run the model on arbitrary scene graphs specified in a simple JSON format. For example to generate images for the scene graphs used in Figure 6 of the paper you can run:

```bash
python scripts/run_model.py \
  --checkpoint checkpoint_with_model.pt \
  --scene_graphs_json scene_graphs/figure_6_sheep.json
```
