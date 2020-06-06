#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import torchvision.transforms as T


# cli library
import click

def decode_segmap(image, source, bgimg, nc=21):
  
  label_colors = np.array([(0, 0, 0),
            
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
 
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
     
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
     
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
    
  rgb = np.stack([r, g, b], axis=2)
  
 
  foreground = cv2.imread(source)


  background = cv2.imread(bgimg)


  foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
  background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
  foreground = cv2.resize(foreground,(r.shape[1],r.shape[0]))
  background = cv2.resize(background,(r.shape[1],r.shape[0]))
  


  foreground = foreground.astype(float)
  background = background.astype(float)

 
  th, alpha = cv2.threshold(np.array(rgb),0,255, cv2.THRESH_BINARY)


  alpha = cv2.GaussianBlur(alpha, (7,7),0)

  
  alpha = alpha.astype(float)/255

  
  foreground = cv2.multiply(alpha, foreground)  
  
 
  background = cv2.multiply(1.0 - alpha, background)  
  
  
  outImage = cv2.add(foreground, background)


  return outImage/255

def segment(net, path, bgimagepath,output_path, show_orig=True, dev='cuda'):
  img = Image.open(path)
  
  if show_orig: plt.imshow(img); plt.axis('off'); plt.show()

  trf = T.Compose([T.Resize(400), 
                   #T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
  inp = trf(img).unsqueeze(0).to(dev)
  out = net.to(dev)(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  
  rgb = decode_segmap(om, path, bgimagepath)

  image = rgb * 255
  image =  Image.fromarray(image.astype(np.uint8))

  image.save(output_path + "/" + str(np.random.randint(10000,9999999999)) + ".png", "PNG")
  


@click.command()
@click.option('--image', default='', help="Path to input image file.")
@click.option('--background', default='', help='Input to input background image file')
@click.option('--output_path', default='', help='Path where file will be saved')
def cli(image,background, output_path):
    if image and background and output_path:
      dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
      segment(dlab, image,background, output_path,show_orig=False)
      click.echo("Successfully!")
      return
    click.echo("Image, output path and background is neccessery params")





