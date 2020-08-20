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

Open the [quickplots notebook](https://github.com/ubcms-xai/XAICoffea/blob/main/quickplots.ipynb) and execute it. 

