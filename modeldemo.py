import torch 
import torchvision
import gradio as gr
from PIL import Image

hidden_layer = torch.nn.Linear(in_features =784, out_features = 128, bias = True)

activation = torch.nn.ReLU()

output_layer = torch.nn.Linear(in_features = 128, out_features = 10, bias = True)

model = torch.nn.Sequential(
  torch.nn.Flatten(), 
  hidden_layer,
  activation,
  output_layer,
)

saved_weights = torch.load(
    'MNIST_MLP.pth',
    weights_only=True,
    map_location="cpu"
    )

model.load_state_dict(saved_weights)

model.eval()

# hey ive done this all already !!!
# the only thing i forgot initially was to load saved_weights

# i need to build a prediction

def predict_image(image):

    if image is None or image.get("composite") is None:
        return ""
    # model wants [1,1,28,28]
    # 1 image, 1 color channel, width, height
    pil_img = Image.fromarray(image["composite"])
    # converts the array from gradio into a pillow image 
    # this is a little confusing but the composite means that it's finished 
    resized_img = pil_img.resize((28,28))
    #self explanatory
    transform = torchvision.transforms.ToTensor()
    img_transform = transform(resized_img)
    # okay this gives me [1, 28, 28]
    # turns into a tensor for the model to see
    # theres something important i needed to add and its another dimension to the tensor
    # [64, 1, 28, 28] is what we'd use in train.py 
    # the 64 is basically the # of images in a batch
    img_transform = img_transform.unsqueeze(0)
    invert_tensor = 1.0 - img_transform

    with torch.no_grad():
        predictions = model(invert_tensor)

    predicted_digits = predictions.argmax().item()
    # i keep forgetting what argmax does
    # okay we have 10 scores 
    # it reminds me of softmaxing but with less words
    # it picks the highest score out of all of them
    return predicted_digits
  

    #print(type(image))
    #print(image.shape)
    #print(resized_img.size)
    #print(img_transform.shape)
    # pillow images dont have a shape so it doesnt work here

    #return 0 # placeholder so it displays a number

demo = gr.Interface(
    fn=predict_image,
    inputs = gr.Sketchpad(
        type="numpy", # we get arrays from this
        # i could do pil but i'd have to change my existing code
        image_mode="L"
    ),
    outputs = gr.Number(),
    live = True
)

demo.launch()