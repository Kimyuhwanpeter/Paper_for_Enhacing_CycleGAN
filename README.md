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
