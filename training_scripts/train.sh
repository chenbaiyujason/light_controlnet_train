export MODEL_DIR="andite/anything-v4.0"
export OUTPUT_DIR="/mnt/disks/data/controlnet_model/control_v1u_sd15_illumination/{timestamp}"
export DATASET_DIR="/mnt/disks/data/consdata/consdeimg/"
export DISK_DIR="/mnt/disks/data/cache/trainlight"
export HUB_MODEL_ID="ioclab/control_v1u_sd15_illumination"

sudo python3 train_controlnet_flax.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path="/mnt/disks/data/controlnet_model/control_v1u_sd15_illumination/control_v1u_sd15_brightness/20230417_120417/4000"  \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATASET_DIR \
 --load_from_disk \
 --cache_dir=$DISK_DIR \
 --validation_image "./conditioning_image_1.jpg" "./conditioning_image_2.jpg" "./conditioning_image_3.jpg" "./conditioning_image_4.jpg" "./conditioning_image_5.jpg" "./conditioning_image_6.jpg" "./conditioning_image_7.jpg" "./conditioning_image_8.jpg" \
 --validation_prompt "a woman sitting at a piano in a dark room with a window behind her and a window behind her, Atey Ghailan, anime art, a painting, neo-romanticism" "a woman with horns and a demon face on her head, with her hands on her face, in front of a demon like background, Ayami Kojima, anime art, a manga drawing, space art" "A girl with her head down, Sailor Moon" "(masterpiece, best quality: 1.4), 1girl,detailed background, white crystal, crysal cluster,long hair,jewelry, earrings, necklace, crown, bride, white hair, halo," "(masterpiece, best quality: 1.4), 1girl,detailed background, white crystal, crysal cluster,long hair,jewelry, earrings, necklace, crown, bride, white hair, halo,"  "(masterpiece, best quality: 1.4), 1girl,detailed background, white crystal, crysal cluster,long hair,jewelry, earrings, necklace, crown, bride, white hair, halo," "(masterpiece, best quality: 1.4), 1girl,detailed background, white crystal, crysal cluster,long hair,jewelry, earrings, necklace, crown, bride, white hair, halo," "(masterpiece, best quality: 1.4), 1girl,detailed background, white crystal, crysal cluster,long hair,jewelry, earrings, necklace, crown, bride, white hair, halo,"  \
 --validation_steps=500 \
 --checkpointing_steps=500 \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=25 \
# --revision="non-ema" \
 --mixed_precision="bf16" \
 --from_pt \
 --num_train_epochs=3 \
 --max_train_steps=55000 \
 --report_to="wandb" \
 --dataloader_num_workers=16 \
 --logging_steps=1 \
 --push_to_hub \
 --hub_model_id=$HUB_MODEL_ID \
 --tracker_project_name="light-controlnet"