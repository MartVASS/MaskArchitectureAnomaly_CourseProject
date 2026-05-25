# Notebooks 

## Semantic segmentation predictions

To visualize semantic segmentation (pixel-wise) prediction from pre-trained model : OPEN visualization_sementic_cityscapes.ipynb
   
REQUIREMENTS : have the cityscapes validation datasets in your google drive : "mydrive/Cityscapes" (TO BE MODIFIED IN YOUR SPECIFIC CASE in preparation code cell.)

   Run the first two cells. This would take 5 minutes, then a request to resart your session pop. Restart your session  and rerun all the cells.
   At the end, you will find the prediction.

## Panoptic segmentation predictions

To visualize panoptic segmentation (mask) prediction from COCO pre-trained model : OPEN visualization_panoptic_cityscapes.ipynb
   
REQUIREMENTS : have the cityscapes validation datasets in your google drive : "mydrive/Cityscapes" (TO BE MODIFIED IN YOUR SPECIFIC CASE in preparation code cell.)

   Run the first two cells. This would take 5 minutes, then a request to resart your session pop. Restart your session  and rerun all the cells.
   At the end, you will find the prediction. 

## Evaluate semantic mIoU of COCO pre-trained model on Cityscapes dataset 

REQUIREMENTS : have the the cityscapes validation datasets in your google drive : "mydrive/Cityscapes" (TO BE MODIFIED IN YOUR SPECIFIC CASE in preparation code cell.)
have the .bin file of the model on your drive : "/content/drive/MyDrive/COCO/eomt_coco.bin"

Run the first two cells. This would take 5 minutes, then a request to resart your session pop. Restart your session  and rerun all the cells.
   Then connect to your wandb account with the API key provided. Then run the next cell who adapt the code to calculate the mIoU using the mapping (COCO -> Cityscapes) between index classes. Then run the last cell and read at the end the mIoU calculated.
   
