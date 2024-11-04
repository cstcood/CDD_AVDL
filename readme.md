# CDD_AVDL
This is the official pytorch implement of CDD_AVDL.

The parameter of Layer fusion should hand compute in different task. The implement of parameter interface and process of training and test
are omitted for commercial protection.

The demo Calling code is shown in `\net\CDD-AVDL.py` `if __name__ == '__main__':`

## Usage

```` 

pip install -r requirements.txt
````
manual download the weight of `vit_deit_small_distilled_patch16_224`  to `.\net\pretrain_models` or just wait  ``timm`` download after calling next command.
````
python .\net\CDD-AVDL.py
````
