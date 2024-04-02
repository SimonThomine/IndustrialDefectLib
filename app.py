from source.defectGenerator import DefectGenerator
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
import numpy as np

def generate_defect_image(image, defect_type,category):
    defGen=DefectGenerator(image.size,dtd_path="samples/dtd/")
    defect,msk=defGen.genDefect(image,[defect_type],category.lower())
    defect=(defect.permute(1,2,0).numpy()*255.0).astype('uint8') 
    msk=(msk.permute(1,2,0).numpy()*255.0).astype('uint8')
    msk = np.concatenate((msk, msk, msk), axis=2)
    return defect, msk

images = {
    "Bottle": Image.open('samples/bottle.png').convert('RGB').resize((1024, 1024)),
    "Cable": Image.open('samples/cable.png').convert('RGB').resize((1024, 1024)),
    "Capsule": Image.open('samples/capsule.png').convert('RGB').resize((1024, 1024)),
    "Carpet": Image.open('samples/carpet.png').convert('RGB').resize((1024, 1024)),
    "Grid": Image.open('samples/grid.png').convert('RGB').resize((1024, 1024)),
    "Hazelnut": Image.open('samples/hazelnut.png').convert('RGB').resize((1024, 1024)),
    "Leather": Image.open('samples/leather.png').convert('RGB').resize((1024, 1024)),
    "Metal Nut": Image.open('samples/metal_nut.png').convert('RGB').resize((1024, 1024)),
    "Pill": Image.open('samples/pill.png').convert('RGB').resize((1024, 1024)),
    "Screw": Image.open('samples/screw.png').convert('RGB').resize((1024, 1024)),
    "Tile": Image.open('samples/tile.png').convert('RGB').resize((1024, 1024)),
    "Toothbrush": Image.open('samples/toothbrush.png').convert('RGB').resize((1024, 1024)),
    "Transistor": Image.open('samples/transistor.png').convert('RGB').resize((1024, 1024)),
    "Wood": Image.open('samples/wood.png').convert('RGB').resize((1024, 1024)),
    "Zipper": Image.open('samples/zipper.png').convert('RGB').resize((1024, 1024))
    
}


def generate_and_display_images(category, defect_type):
    base_image = images[category]
    img_with_defect, defect_mask = generate_defect_image(base_image, defect_type,category)
    return np.array(base_image), img_with_defect, defect_mask

# Components 
with gr.Blocks(css="style.css") as demo:
    gr.HTML(
        "<h1><center> &#127981; MVTEC AD Defect Generator &#127981; </center></h1>" +
        "<p><center><a href='https://github.com/SimonThomine/IndustrialDefectLib'>https://github.com/SimonThomine/IndustrialDefectLib</a></center></p>"
    )
    with gr.Group():
        with gr.Row():
            category_input = gr.Dropdown(label="Select object", choices=list(images.keys()),value="Bottle")
            defect_type_input = gr.Dropdown(label="Select type of defect", choices=["blurred", "nsa","structural", "textural" ],value="nsa")
            submit = gr.Button(
                scale=1,
                variant='primary'
            )        
        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                gr.HTML("<h1><center> Base </center></h1>")
                base_image_output = gr.Image("Base", type="numpy")
                
            with gr.Column(scale=1, min_width=400):
                gr.HTML("<h1><center> Mask </center></h1>")
                mask_output = gr.Image("Mask", type="numpy")
                
            with gr.Column(scale=1, min_width=400):
                gr.HTML("<h1><center> Defect </center></h1>")
                defect_image_output = gr.Image("Defect", type="numpy")
                
    submit.click(
        fn=generate_and_display_images,
        inputs=[category_input, defect_type_input],
        outputs=[base_image_output, defect_image_output,mask_output],
    )
    
demo.launch()