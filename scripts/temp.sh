gcloud alpha compute tpus tpu-vm ssh funk-fusion-node --zone us-central2-b --project hf-flax

pip install jsonlines numpy requests datasets pillow

cd ~/controlnet-training
cd jax-controlnet-sprint/dataset_tools

cd ~/controlnet-training/jax-controlnet-sprint/dataset_tools
cd ~/controlnet-training/jax-controlnet-sprint/training_scripts
source ~/.venv/bin/activate
sudo bash train.sh
jax-smi

cd /mnt/disks/persist/data/
sudo chmod -R 777 /mnt/

git clone https://huggingface.co/datasets/laion/laion2B-en-aesthetic/
cd /mnt/disks/persist/laion2B-en-aesthetic

python laion_download.py
cd /mnt/disks/persist/images/

# 统计文件夹文件大小
du -sh ./

# 统计文件夹文件个数
ls -l | grep "^-" | wc -l

tmux attach -t 1

cd /mnt/disks/persist/datasets/3m/

rm -r /mnt/disks/persist/datasets/3m

cp ./data.py /mnt/disks/persist/datasets/3m/3m.py

sudo killall -u ciaochaos