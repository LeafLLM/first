import torch 
import torchvision
import gradio as gr
from PIL import Image
from PIL import Image, ImageOps

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
    # also the convert means converting to grayscale 
    # this is a little confusing but the composite means that it's finished 
    resized_img = pil_img.resize((28,28))
    invert_img = ImageOps.invert(pil_img.convert("L"))
    #alright to tackle the resizing issue i need to crop around the drawn image
    # bbox finds the where to crop
    #crop actually crops the img based on the bbox
    bbox = invert_img.getbbox()
    if bbox is not None:
        cropped_img = invert_img.crop(bbox)

        print("Bound box:", bbox)
        print("cropped box:", cropped_img)

        width, height = cropped_img.size

        # size returns the largest number
        # also cropped_img returns (a, b)
        #   so we're giving these values a label i think

        scale = 20/max(width, height)
        # image wont be perfectly square
        # we want to scale down to 20 
        new_width = max(1,round(width*scale))
        # now we use the scale!
        new_height = max(1, round(height*scale))

        resized_img = cropped_img.resize((new_width, new_height))
        paste_img = Image.new("L",(28,28), 0)
        # grayscale(mode), size, color 
        #now we need to center the thing

        x = (28-new_width) // 2
        y = (28-new_height) // 2

        paste_img.paste(resized_img, (x,y))
        # asks for image, and the coordinates


        

    #self explanatory
    transform = torchvision.transforms.ToTensor()
    img_transform = transform(paste_img)
    # okay this gives me [1, 28, 28]
    # turns into a tensor for the model to see
    # theres something important i needed to add and its another dimension to the tensor
    # [64, 1, 28, 28] is what we'd use in train.py 
    # the 64 is basically the # of images in a batch

    img_transform = img_transform.unsqueeze(0)
    #invert_tensor = 1.0 - img_transform

    print( "resized:", resized_img.size)
    print("final:", paste_img.size)
    print ("tensor:", img_transform.shape)

    with torch.no_grad():
        predictions = model(img_transform)

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
        image_mode="L",
        brush = gr.Brush(
            default_size = 20,
            default_color="black",
            color_mode="fixed"
                         )
    ),
    outputs = gr.Number(),
    live = True
)

demo.launch()