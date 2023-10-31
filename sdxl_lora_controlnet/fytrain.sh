accelerate launch --num_cpu_threads_per_process 1 train_network.py 
    --pretrained_model_name_or_path=/root/autodl-tmp/sdxlFixedvaeFp16Remove_baseFxiedVaeV2Fp16.safetensors
    --output_dir=<训练过程中的模型输出文件夹>  
    --output_name=<训练模型输出时的文件名> 
    --save_model_as=safetensors 
    --prior_loss_weight=1.0 
    --max_train_steps=400 
    --learning_rate=1e-4 
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
    --save_every_n_epochs=1 
    --network_module=networks.lora




python sdxl_train_control_net_lllite.py --config_file /root/autodl-tmp/sdxl_lora/tile_controlnet/train.toml 