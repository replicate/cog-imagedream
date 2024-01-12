# Cog wrapper for ImageDream

A Cog wrapper for ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation. See the [paper](https://arxiv.org/abs/2312.02201), original project [page](https://image-dream.github.io/) and Replicate [demo](https://replicate.com/adirik/imagedream) for details. 


## API Usage

You need to have Cog and Docker installed to run this model locally. To use ImageDream, you need to upload an image or provide a text prompt for the desired object to be generated. The output will be a textured 3D object file in .glb format. Depending on the parameters you set, the 3D model will be generated in 1h-2h on an A40 GPU.

To build the docker image with cog and run a prediction:
```bash
cog predict -i image=@astronaut.png
```

To start a server and send requests to your locally or remotely deployed API:
```bash
cog run -p 5000 python -m cog.server.http
```

To use ImageDream, simply enter a text description and corresponding image of 3D asset you want to generate. The input arguments are as follows: 
- **image:** Image of an object to generate a 3D object from. The object should be placed in the center and must not be too small/big in the image.  
- **prompt:** Short text description of the 3D object to generate.  
- **negative_prompt:** Short text description of the 3D object to not generate.  
- **guidance_scale:** The higher the value, the more similar the generated 3D object will be to the inputs.  
- **shading:** If set to True, the texture of the generated 3D object will be better, but the generation takes ~2h. If set to False, the texture of the generated 3D object will be worse, but the generation takes ~1h.  
- **num_steps:** Number of training steps. Strongly advised to keep the default value for optimal results.  
- **seed:** Seed for reproducibility, default value is None. Use default value for random seed. Set to an arbitrary value for deterministic generation.

## References
```
@article{wang2023imagedream,
  title={ImageDream: Image-Prompt Multi-view Diffusion for 3D Generation},
  author={Wang, Peng and Shi, Yichun},
  journal={arXiv preprint arXiv:2312.02201},
  year={2023}
}
```
