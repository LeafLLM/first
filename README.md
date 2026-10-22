# Classification Neural Network

this project is supposed to get me started with machine learning!!! 
it's trained off the MNIST data set (basically a data of a bunch of handwritten numbers)
the neural network is supposed to classify these numbers 0-9 

so far its helped me learn a lot about activation & loss functions. 
in this you'll see that almost every line has a piece of code, or a bunch of print functions to test if my values were reasonable.
this is mostly just a learning step for me.

for this to work you'll have to have python installed, and read the installation guide for pytorch 

right now im trying to figure out how to let my neural network go through all the data instead of running 1 epoch (which is just passing a set amount of data through the model)
i think it also makes a new model instead of running off the previous one, so thats a problem too if i want to train it

COOL!!! i got my model to run different epochs and it can load and save itself!!! my model also prints its accuracy :3

# Running the Program

right now this only runs locally 
you'd need python, torch, torchvision and gradio installed in order to run this
in a terminal you'd run python modeldemo.py
i want to work on being able to host this publicly but ill get there
i also did a little training and the model accuracy got to 98.11%!
also train.py runs only on with cuda, this is because of my workflow rn....

# Issues?

i noticed that this was somewhat inaccurate, and it might be for two things
1. the brush size is thicker than the ones used in the MNIST dataset
2. resizing probably didn't center the images like the MNIST numbers
3. i write weird

i think with my next commits it'll be trying to unpack the resizing issue
also hopefully i can figure out how to make the interface less finicky
i hypothesize that resizing the data messes with the model since its trained on numbers that are dead center
i dont write like that