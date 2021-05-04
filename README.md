# XAICoffea
XAI modules for PythiaGenJets ntuples with Coffea

## (Optional) Create a test file with `PythiaGenJets`

The input files are made with [this](https://github.com/rappoccio/PythiaGenJets) package. To create a test example:

```
bash ./runDockerCommandLine.sh srappoccio/pythia-gen-jets:latest
make pythia2root
./pythia2root zz_flatter.cfg test.root 100
```

The file `test.root` is attached here for reference. 

## Run on the test file

Start [this docker image](https://hub.docker.com/repository/docker/srappoccio/ubccr-cms). 

```
wget https://raw.githubusercontent.com/rappoccio/ubccr-cms/master/runUBCCRCMS.sh
chmod a+x runUBCCRCMS.sh
./runUBCCRCMS.sh 8888 srappoccio/ubccr-cms:latest
```

Within the docker image:

```
jupyter notebook --ip 0.0.0.0 --no-browser 
```

Then point your webbrowser to the instance (if you are local, try [localhost](https://localhost:8888)). 

Open the [quickplots notebook](https://github.com/ubcms-xai/XAICoffea/blob/main/quickplots.ipynb) within the notebook and execute it.

## Machine Learning Studies - Common Files

**[Network Builder](https://github.com/ubcms-xai/XAICoffea/blob/main/python/networkBuilder.py)**: Builds common network architectures

**[Analysis Helper](https://github.com/ubcms-xai/XAICoffea/blob/main/python/analysisHelper.py)**: Defines LRP functions and relevance bar plot function


## Toy Model Studies

### Preprocssing
We start by generating toy model data. The z and theta distibutions for signal are derived from a normal distribution and the z and theta for the background sample are derived from an exponential disttibution. To run this scipt, we first connect to UBCCR image that supports coffea

```
sudo ./runUBCCRCMS.sh srappoccio/ubccr-cms:latest
```

and once you have a port, start the jupyter notebook:

```
jupyter notebook --ip 0.0.0.0 --no-browser 
```
To produce training and testing samples run the notebook - [makeJetImages.ipynb](https://github.com/ubcms-xai/XAICoffea/blob/main/ToyModel/makeJetImages.ipynb)
This will produce seperate files for training and testing samples that inclues toy signal and background events.

### Neural Networks

When training, testing, or analyzing the networks with LRP, we need to use the innvestigate_tensorflow docker image. (Using tensorflow version 1.)

```
sudo ./runUBCCRCMS.sh srappoccio/innvestigate_tensorflow:latest
```

Within the docker image:

```
jupyter notebook --ip 0.0.0.0 --no-browser 
```

#### 2D CNN

For 2D CNN we train 2 different kinds of models: one with image as imput and another with image ans XAUGs.
Both can be built, initialised, trained and evaluated using [CNN_2D.ipynb](https://github.com/ubcms-xai/XAICoffea/blob/main/ToyModel/CNN_2D.ipynb). This sript also produces some elementary plots to visualise the model performance.

Next, we can run LRP on the trained model to learn the relevance of an input in the NN decision. Depending on the model we are interested in we have two seperate scrpits:
  * [Running LRP on Toy model with Image as input](https://github.com/ubcms-xai/XAICoffea/blob/main/ToyModel/LRP_toy_imageOnly.ipynb)
  * [Running LRP on Toy model with Image + XAUGs as inputs](https://github.com/ubcms-xai/XAICoffea/blob/main/ToyModel/LRP_toy_XAUG.ipynb)
 

Both scrpits run LRP on the models and produce plots to visualize the relevance of each input variable.

## Pythia Model Studies

### Preprocessing

Start by preprocessing the pythia files. (Preprocessing explained in detail in [our paper](https://arxiv.org/abs/2011.13466) )

First, connect to the UBCCRCMS docker image

```
sudo ./runUBCCRCMS.sh srappoccio/ubccr-cms:latest
```

Within the docker image:

```
jupyter notebook --ip 0.0.0.0 --no-browser 
```

Run [Showjets.ipynb](https://github.com/ubcms-xai/XAICoffea/blob/main/ShowJets.ipynb)

This will:

* Select the leading jet and keep its leading 20 constituents
* Select the variables of interest
* Rotate, center, and scale the image
* Save npz files

Run [CNN_Data_preprocess_all.ipynb](https://github.com/ubcms-xai/XAICoffea/blob/main/CNN_Data_preprocess_all.ipynb)

This will:

* Normalize the single-variable inputs
* Create test and train datasets for signal (Z to bb) and background (QCD)

### Neural Networks

When training, testing, or analyzing the networks with LRP, we need to use the innvestigate_tensorflow docker image. (Using tensorflow version 1.)

```
sudo ./runUBCCRCMS.sh srappoccio/innvestigate_tensorflow:latest
```

Within the docker image:

```
jupyter notebook --ip 0.0.0.0 --no-browser 
```

#### 2D CNN

 * Plot the inputs: [CNN_Data_plotting.ipynb](https://github.com/ubcms-xai/XAICoffea/blob/main/CNN_2D/CNN_Data_plotting.ipynb)
 * Build the models: [CNN_fit_model.ipynb](https://github.com/ubcms-xai/XAICoffea/blob/main/CNN_2D/CNN_fit_model.ipynb)
 * Test the models and analyze with LRP: [CNN_Analyze.ipynb](https://github.com/ubcms-xai/XAICoffea/blob/main/CNN_2D/CNN_Analyze.ipynb)

#### 1D CNN

## Plotting
