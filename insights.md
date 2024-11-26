## Tested Examples

### Content/Style tradeoff

Changing style weight gives you less or more style on the final image, assuming you keep the content weight constant. <br/>
I did increments of 10 here for style weight (1e1, 1e2, 1e3, 1e4), while keeping content weight at constant 1e5, and I used random image as initialization image. 

<p align="center">
<img src="data/examples/style-tradeoff/figures_vg_starry_night_o_lbfgs_i_random_h_352_m_vgg19_cw_100000.0_sw_10.0_tv_1.0_resized.jpg" width="200px">
<img src="data/examples/style-tradeoff/figures_vg_starry_night_o_lbfgs_i_random_h_352_m_vgg19_cw_100000.0_sw_100.0_tv_1.0_resized.jpg" width="200px">
<img src="data/examples/style-tradeoff/figures_vg_starry_night_o_lbfgs_i_random_h_352_m_vgg19_cw_100000.0_sw_1000.0_tv_1.0_resized.jpg" width="200px">
<img src="data/examples/style-tradeoff/figures_vg_starry_night_o_lbfgs_i_random_h_352_m_vgg19_cw_100000.0_sw_10000.0_tv_1.0_resized.jpg" width="200px">
</p>

### Impact of total variation (tv) loss

Rarely explained, the total variation loss i.e. it's corresponding weight controls the smoothness of the image. <br/>
I also did increments of 10 here (1e1, 1e4, 1e5, 1e6) and I used content image as initialization image.

<p align="center">
<img src="data/examples/tv-tradeoff/figures_candy_o_lbfgs_i_content_h_350_m_vgg19_cw_100000.0_sw_30000.0_tv_10.0_resized.jpg" width="200px">
<img src="data/examples/tv-tradeoff/figures_candy_o_lbfgs_i_content_h_350_m_vgg19_cw_100000.0_sw_30000.0_tv_10000.0_resized.jpg" width="200px">
<img src="data/examples/tv-tradeoff/figures_candy_o_lbfgs_i_content_h_350_m_vgg19_cw_100000.0_sw_30000.0_tv_100000.0_resized.jpg" width="200px">
<img src="data/examples/tv-tradeoff/figures_candy_o_lbfgs_i_content_h_350_m_vgg19_cw_100000.0_sw_30000.0_tv_1000000.0_resized.jpg" width="200px">
</p>

### Optimization initialization

Starting with different initialization images: noise (white or gaussian), content and style leads to different results. <br/>
Empirically content image gives the best results as explored in [this research paper](https://arxiv.org/pdf/1602.07188.pdf) also. <br/>
Here you can see results for content, random and style initialization in that order (left to right):

<p align="center">
<img src="data/examples/init_methods/golden_gate_vg_la_cafe_o_lbfgs_i_content_h_500_m_vgg19_cw_100000.0_sw_30000.0_tv_1.0_resized.jpg" width="270px">
<img src="data/examples/init_methods/golden_gate_vg_la_cafe_o_lbfgs_i_random_h_500_m_vgg19_cw_100000.0_sw_1000.0_tv_1.0_resized.jpg" width="270px">
<img src="data/examples/init_methods/golden_gate_vg_la_cafe_o_lbfgs_i_style_h_500_m_vgg19_cw_100000.0_sw_10.0_tv_0.1_resized.jpg" width="270px">
</p>

You can also see that with style initialization we had some content from the artwork leaking directly into our output.

### Famous "Figure 3" reconstruction

Finally if I haven't included this portion you couldn't say that I've successfully reproduced the [original paper]((https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)) (laughs in Python):

<p align="center">
<img src="data/examples/gatys_reconstruction/tubingen.jpg" width="300px">
<img src="data/examples/gatys_reconstruction/tubingen_shipwreck_o_lbfgs_i_random_h_400_m_vgg19_cw_100000.0_sw_200.0_tv_1.0_resized.jpg" width="300px">
<img src="data/examples/gatys_reconstruction/tubingen_starry-night_o_lbfgs_i_content_h_400_m_vgg19_cw_100000.0_sw_30000.0_tv_1.0.jpg" width="300px">

<img src="data/examples/gatys_reconstruction/tubingen_the_scream_o_lbfgs_i_random_h_400_m_vgg19_cw_100000.0_sw_300.0_tv_1.0.jpg" width="300px">
<img src="data/examples/gatys_reconstruction/tubingen_seated-nude_o_lbfgs_i_random_h_400_m_vgg19_cw_100000.0_sw_2000.0_tv_1.0.jpg" width="300px">
<img src="data/examples/gatys_reconstruction/tubingen_kandinsky_o_lbfgs_i_content_h_400_m_vgg19_cw_100000.0_sw_30000.0_tv_1.0.jpg" width="300px">
</p>

I haven't give it much effort results can be much nicer.

### Content reconstruction

If we only use the content (perceptual) loss and try to minimize that objective function this is what we get (starting from noise):

<p align="center">
<img src="data/examples/content_reconstruction/0000.jpg" width="200px">
<img src="data/examples/content_reconstruction/0026.jpg" width="200px">
<img src="data/examples/content_reconstruction/0070.jpg" width="200px">
<img src="data/examples/content_reconstruction/0509.jpg" width="200px">
</p>

In steps 0, 26, 70 and 509 of the L-BFGS numerical optimizer, using layer relu3_1 for content representation.<br/> 

### Style reconstruction

We can do the same thing for style (on the left is the original art image "Candy") starting from noise:

<p align="center">
<img src="data/examples/style_reconstruction/candy.jpg" width="200px">
<img src="data/examples/style_reconstruction/0045.jpg" width="200px">
<img src="data/examples/style_reconstruction/0129.jpg" width="200px">
<img src="data/examples/style_reconstruction/0510.jpg" width="200px">
</p>

In steps 45, 129 and 510 of the L-BFGS using layers relu1_1, relu2_1, relu3_1, relu4_1 and relu5_1 for style representation.