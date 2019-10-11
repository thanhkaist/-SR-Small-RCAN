EE838A Homework1
============

### How to run the code

1. Copy data to the folder SR_data such that we have:
- SR_data/train : train data
- SR_data/benchmark : benchmark data
2. Install dependency with Python 3.6.9
```pip install -r requirement.txt```
3. Train network with command below
```python main.py```
4. After train is completed, test the network with command below
```python test.py```

All the result will be store in **result** folder. The result includes:
- log.txt : log file
- model: model checkpoints
- SR_img* : super resolution image for test set

