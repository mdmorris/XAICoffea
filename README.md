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

Run the training

#### 1D CNN

## Plotting
