import gradio as gr 

def double_number(num):
    return num * 2
#this is like public static double [name]([variable name]) {
# return [variable name]*2; 
#} 
#i miss coding in java 

demo = gr.Interface(
    fn=double_number, #this calls our function 
    inputs=gr.Number(), #you can input a number
    outputs=gr.Number() # displays a number
)
demo.launch()

