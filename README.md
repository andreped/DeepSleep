# DeepSleep

## Setup

When using this framework, it is a good idea to setup a virtual environment:
```
virtualenv -ppython3 venv --clear
source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.7.9, on Win10, macOS, and Ubuntu Linux operating systems.

Note that to activate the virtual environment on Windows instead run `./venv/Scripts/activate`.

## Usage

To train a model, simply run:
```
python main.py
```

The script supports multiple arguments. To see supported arguments, run `python main.py -h`.


## Acknowledgements

The mobile app was developed using [Flutter](https://github.com/flutter/flutter), which is a framework developed by Google.
For the app, the following _open_ packages were used (either MIT, BSD-2, or BSD-3 licensed):
* [wakelock](https://pub.dev/packages/wakelock)
* [tflite_flutter](https://pub.dev/packages/tflite_flutter)
* [watch_connectivity](https://pub.dev/packages/watch_connectivity)
* [watch_connectivity_garmin](https://pub.dev/packages/watch_connectivity_garmin)
