from processors.ddp_mix_processor import DDPMixProcessor



# python -m torch.distributed.launch --nproc_per_node=1 main.py
if __name__ == '__main__':
    processor = DDPMixProcessor(cfg_path="configs/faster_rcnn_coco.yml")
    processor.run()


