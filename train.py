import torch 
import torchvision 

train_data = torchvision.datasets.MNIST( 
    root='data',
    train=True,
    download=True,
    transform = torchvision.transforms.ToTensor() # turns the data into tensor
    # think of it as a matrix, but theres more dimensions to it.
    # so ex. (width, hei)
)

test_data = torchvision.datasets.MNIST(
    root ='data',
    train=False,
    download=True,
    transform = torchvision.transforms.ToTensor()
)



image = train_data[0][0]  
# first image in the set
# when you print train_data[0], you get a tuple, aka it holds different things
# that one in particular returns the image + label for the first image in the set
#label = train_data[0][1] 
# returns the label for the first image, in this case it should be 5

flat_image = image.flatten()
# flattening an image basically means we take (1,28,28) and multiply each number
# This is why we get 784 when we print flat_image

hidden_layer = torch.nn.Linear(in_features =784, out_features = 128, bias = True)
# hey thats my first layer of my NN! 
# it has 784 input features and it outputs 128 new values. 
# it'll have bias/weights, i think it'll learn these through training

hidden_output = hidden_layer(flat_image)
# ok we use the hidden layer function, which is my 128 outputs, and we put the condensed image through this 
# it gets stored in hidden_output

activation = torch.nn.ReLU()
# this creates a relu layer, aka introducing nonlinearity (all that stuff from earlier)
# yay!! lets add complexity!!

active_output = activation(hidden_output)
# now we're calling the relu layer and it's using the numbers from hidden_output

output_layer = torch.nn.Linear(in_features = 128, out_features = 10, bias = True)
# this should be self explanatory. you get another layer and it turns them into 10 numbers the model picks from basically...
result = output_layer(active_output)

# print (type(image)) # this prints the type of the variable image "What object is this?"
#print (image.shape) # this prints the dimensions of the variable image "What is its dimensions?"
#print (flat_image.shape) 
#print(hidden_output)
#print(active_output)
#print(result.shape)
#print(result.argmax())


#actual model yaaay!!!

model = torch.nn.Sequential(
  torch.nn.Flatten(), 
  hidden_layer,
  activation,
  output_layer,
)

device = torch.device("cuda")
model = model.to(device)


saved_weights = torch.load(
    "MNIST_MLP.pth",
    map_location="cuda",
    weights_only=True
)

model.load_state_dict(saved_weights)

# kinda the same thing but it does all the steps above automatically

#prediction = model(image)
# yeah its using the sequence to do stuff with an image

loss_fn = torch.nn.CrossEntropyLoss()
# hey its that funny thing called loss!!!
# its basically telling how wrong yo shit is
# cross entropy loss used here because we classifying shit

#target = torch.tensor([label])
# basically the correct number, or the number its trying to predict
# without the brackets it just throws "tensor(5)"
# for CE loss we need it to be tensor ([5]) cause then it has a dimension of 1
# and prediction is two dimensional since if you print it you get torch.Size[1,10]
# basically that dimension tells us theres 1 image, 10 different scores

# loss_output = loss_fn(prediction, target)
# This only passes one image. And the target is only 

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr = 0.0001
    )
#this is basically the thing that updates the model. LR stands for learning rate
# Actually for this model, I don't know what happens if you increase the LR more.
# I only know if you set it too high, the LR causes the model to overshoot all the time
# Then learning gets harder

#weights_before = hidden_layer.weight.detach().clone()
# used this to double check if our weights got updated, it's not updated and it's a copy of that value

#optimizer.zero_grad()
# clears previous gradients

#loss_output.backward()
# ye get your gradients 

#optimizer.step()
# this just updates it 

#print(torch.equal(weights_before, hidden_layer.weight))
# comparison to check if step worked

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True
)
# yay now we can batch our data and train our model with it!!!

#print(len(train_loader))


for i in range(10):
    # now we can do multiple epoches! 
    # also 5 (it can be anything else honestly) is a hyperparameter aswell

    total_loss = 0
    # self explanatory here
    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
    
        loss_outputs = loss_fn(predictions, labels)
       
        total_loss += loss_outputs.item()

        optimizer.zero_grad()
        loss_outputs.backward()
        optimizer.step()
    # okay lowkey idk if i keep these inside our outside the loop,
    #  probably inside

   # print("Before:", loss_outputs.item())

   # predictions_after = model(images)
    #loss_after = loss_fn(predictions_after, labels)

    #print("After:", loss_after.item())
   # print(predictions.shape)

    average_batch_loss = total_loss / len(train_loader)

   # print(average_batch_loss)

print(f"Epoch {i+1}: {average_batch_loss:.4f}")
    # also apparently this is a f string.
    # i could just do "epoch:" + str(i +1) + yadadada but this is easy to read
    #putting the function:.4f rounds this to the 4th place!!

test_loader = torch.utils.data.DataLoader(
    dataset = test_data,
    batch_size=64,
    shuffle = False
)

total_correct = 0
model.eval()

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        testing_predictions = model(images)
        predicted_digits = testing_predictions.argmax(dim=1)
        total_correct += ((predicted_digits == labels).sum().item())
accuracy = (total_correct/ len(test_data))*100
print(accuracy)

torch.save(model.state_dict(), 'MNIST_MLP.pth')