# Catching the Blackdog Easily: A Convenient Depression Diagnosis Method based on Audio-Visual Deep Learning
This is the official pytorch implement of "Catching the Blackdog Easily: A Convenient Depression Diagnosis Method based on Audio-Visual Deep Learning".

The parameter of Layer fusion should hand compute in different task. The implement of parameter interface and process of training and test
are omitted for commercial protection.

The demo Calling code is shown in `\net\CDD-ACDL.py` `if __name__ == '__main__':`

## Usage

```` 

pip install -r requirements.txt
````
manual download the weight of `vit_deit_small_distilled_patch16_224`  to `.\net\pretrain_models` or just wait  ``timm`` download after calling next command.
````
python .\net\CDD-ACDL.py
````
