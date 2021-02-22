# MobileNet_V2
Unofficial implementation of MobileNet-V2 in PyTorch.

Reference : <a href="https://arxiv.org/pdf/1801.04381.pdf">https://arxiv.org/pdf/1801.04381.pdf</a>
<br>
<section>

MobileNet-V2 addresses the challenges of using deep learning models in resource constraints environments, e.g., mobile devices and embedded systems. The main idea behind MobileNet-V2 is to replace most of the regular convolutional layers in a conventional CNN model with <b>Inverted Residual blocks</b>. These blocks are made of <b>depth-wise</b> convolutions (with the kernel size of 3 * 3), <b>point-wise</b> convolutions (with the kernel size of 1 * 1), both equipped with <b>non-linear activations</b>, and a final <b>point-wise</b> convolution with linear mapping. The figure below depicts the mechanism of depth-wise, and point-wise convolutional layers, as well as inverted residual blocks.

<br>
<p><b>PS-1</b>: In the original paper, the activation function for non-linear transformation is <b>ReLU-6</b>. In this implementation, I have replaced ReLU-6 with the regular ReLU.</p>
<p><b>PS-2</b>: In the original paper, the last layer is a regular convolutional layer with <b>1000 channels</b> (number of ImageNet classes), and <b>(1,1)</b> spatial size. In this implementation, <b>a combination of dropout(0.2), and a linear layer</b> has been used instead of the mentioned CNN layer (ispired by the <a href="https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py">PyTorch</a> implementation)</p>
</section>
<br>
<section>
  <h2>Architecture</h2>
  <p>The following table is the architecture of MobileNet-V2. </p>
  <br>
  <table>
  <tr>
    <th>Input size</th>
    <th>layer/module</th> 
    <th>expansion rate (t)</th>
    <th># out-channels (c)</th>
    <th># of layer/module (n)</th>
    <th>stride (s)</th>
  </tr>
  <tr>
    <td>(3,224,224)</td>
    <td>Conv2d</td> 
    <td>-</td>
    <td>32</td>
    <td>1</td>
    <td>2</td>
  </tr>
   <tr>
    <td>(32,112,112)</td>
    <td>bottleneck</td> 
    <td>1</td>
    <td>16</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td>(16,112,112)</td>
    <td>bottleneck</td> 
    <td>6</td>
    <td>24</td>
    <td>2</td>
    <td>2</td>
  </tr>
   <tr>
    <td>(24,56,56)</td>
    <td>bottleneck</td> 
    <td>6</td>
    <td>32</td>
    <td align="center">3</td>
    <td>2</td>
  </tr>
   <tr>
    <td>(32,28,28)</td>
    <td>bottleneck</td> 
    <td>6</td>
    <td>64</td>
    <td>4</td>
    <td>2</td>
  </tr>
   <tr>
    <td>(64,14,14)</td>
    <td>bottleneck</td> 
    <td>6</td>
    <td>96</td>
    <td>3</td>
    <td>1</td>
  </tr>
   <tr>
    <td>(96,14,14)</td>
    <td>bottleneck</td> 
    <td>6</td>
    <td>160</td>
    <td>3</td>
    <td>2</td>
  </tr>
   <tr>
    <td>(160,7,7)</td>
    <td>bottleneck</td> 
    <td>6</td>
    <td>320</td>
    <td>1</td>
    <td>1</td>
  </tr>
   <tr>
    <td>(320,7,7)</td>
    <td>Conv2d (1x1)</td> 
    <td>-</td>
    <td>1280</td>
    <td>1</td>
    <td>1</td>
  </tr>
   <tr>
    <td>(1280,7,7)</td>
    <td>avgpool (7x7)</td> 
    <td>-</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
   <tr>
    <td>(1280,1,1)</td>
    <td>Conv2d (1x1)</td> 
    <td>-</td>
    <td>num_classes (1000)</td>
    <td>-</td>
    <td>-</td>
  </tr>

</table>
  </section>
