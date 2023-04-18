from diffusers import ControlNetModel

model = ControlNetModel.from_pretrained("diffusion_flax_model.msgpack", from_flax=True)