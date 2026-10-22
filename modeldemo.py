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
    # model wants [1,1,28,28]
    # 1 image, 1 color channel, width, height
    pil_img = Image.fromarray(image)
    # converts the array from gradio into a pillow image 
    resized_img = pil_img.resize((28,28))
    #self explanatory
    transform = torchvision.transforms.ToTensor()
    img_transform = transform(resized_img)
    # turns into a tensor for the model to see

    print(type(image))
    print(image.shape)
    print(resized_img.size)
    print(img_transform.shape)
    # pillow images dont have a shape so it doesnt work here

    return 0 # placeholder so it displays a number

demo = gr.Interface(
    fn=predict_image,
    inputs = gr.Image(image_mode="L"),
    outputs = gr.Number()
)

demo.launch()