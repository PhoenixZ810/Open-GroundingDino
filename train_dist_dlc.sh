conda init bash && . /root/.bashrc
conda activate /mnt/data/mmperc/zhaoxiangyu/env/grounding
cd /mnt/data/mmperc/zhaoxiangyu/open-groundingdino
python -V
export PYTHONPATH=$PYTHONPATH:/mnt/data/mmperc/zhaoxiangyu/env/grounding/
export HOME=/mnt/data/mmperc/zhaoxiangyu
export NCCL_SOCKET_IFNAME=eth0
python -m torch.distributed.launch \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT \
--nproc_per_node=8 \
--nnodes=$WORLD_SIZE \
--node_rank=$RANK \
main.py \
--output_dir "try_log" \
-c "config/cfg_odvg.py" \
--datasets "config/datasets_try.json"  \
--options text_encoder_type=checkpoints/bert-base-uncased
