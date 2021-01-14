# Paper_for_Enhaced_CycleGAN with untrained age
<br/>

* Paper: ["Enhanced Cycle Generative Adversarial Network for Generating Face Images of Untrained Races and Ages for Age Estimation"](https://ieeexplore.ieee.org/document/9311721)
<br/>

## Implementation
* tensorflow >= 2.0
* python >= 3.5.0 
* Ubuntu >= 18.04
<br/>

## Train(FLAGS)
* "A_image_txt": Read A information using the text files

  * text: image and age label list

    * | A_image.txt                                               |
      | --------------------------------------------------------- | 
      | image1.jpg 16<br/>image2.jpg 20<br/>image3.jpg 25<br/>... |
* "B_image_txt": Read B information using the text files

  * text: IMage and label list (same form as "A_image_txt")
  
    * | B_image.txt                                                |
      | ---------------------------------------------------------- | 
      | image5.jpg 16<br/>image10.jpg 20<br/>image3.jpg 25<br/>... |
      
  * Precautions - **The age difference between A and B images must be constant.**
  
* "A_image_path": A image directory for training (A real data)
* "B_image_path": B image directory for training (B real data)
* "Number_A_image": Number of A images (**The amount of the images must same as B**)
* "Number_B_image": Number of B images (**The amount of the images must same as A**)
<br/>

## Test(FLAGS)
* "train": False (if not "test", False)
* "A_test_img": A image directory for testing (A real data)
* "B_test_img": B image directory for testing (B real data)
* "A_n_test": Number of A images (**The amount of the images must same as B**)
* "B_n_test": Number of B images (**The amount of the images must same as A**)
* "A_test_output": Path for Generating image (A to B)
* "B_test_output": Path for Generating image (B to A)

## Model
* Proposed model
<br/>

![image-20210114111521140](https://github.com/Kimyuhwanpeter/Paper_for_Enhacing_CycleGAN/blob/main/Proposed_model.JPG)
<br/>

* Proposed 3D-one hot encoding using gray scale image
<br/>

![image-20210114111948668](https://github.com/Kimyuhwanpeter/Paper_for_Enhacing_CycleGAN/blob/main/3D_one_hot.JPG)
