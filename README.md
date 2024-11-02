# augment-images
A simple image augmentation program with support for various augmentations. Small project made for the Fundamentals of Computer Vision lab during my first year of Machine Learning master studies at Polytechnic University of Timișoara.

## Running the program

Into your Terminal, run the following command:

`python augment.py config_file.txt`

Replace `config_file.txt` with the path to the desired configuration file.

## Format of configuration file 

The configuration file uses a simple, line-based format:

```
operation1,param1,param2;operation2,param1
operation3,param1
```


Each line represents a set of operations applied sequentially. Multiple operations on a single line are separated by semicolons. Operation parameters are comma-separated.

## Documentation

Read more about this project at the [documentation](https://github.com/efrem-upt/augment-images/blob/main/docs/FCV_Project.pdf).

## License

[MIT](https://choosealicense.com/licenses/mit/)

