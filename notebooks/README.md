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
## Evaluate the mIoU of the Cityscapes pre-trained model on Cityscapes dataset.

REQUIREMENTS : have the cityscapes validation datasets in your google drive : "mydrive/Cityscapes" (TO BE MODIFIED IN YOUR SPECIFIC CASE in preparation code cell.)
have the .bin file of the model on your drive : "/content/drive/MyDrive/Cityscapes/eomt_cityscapes.bin"

To evaluate this model, OPEN Eval_eomt_semantic.ipynb. 
Run the first two cells. This would take 5 minutes, then a request to resart your session pop. Restart your session  and rerun all the cells.
   Then connect to your wandb account with the API key provided. Then run the last cell and read at the end the mIoU calculated.


## Evaluate semantic mIoU of COCO pre-trained model on Cityscapes dataset 

REQUIREMENTS : have the the cityscapes validation datasets in your google drive : "mydrive/Cityscapes" (TO BE MODIFIED IN YOUR SPECIFIC CASE in preparation code cell.)
have the .bin file of the model on your drive : "/content/drive/MyDrive/COCO/eomt_coco.bin"

To evaluate this model, OPEN Evaluation_comparison.ipynb. 
Run the first two cells. This would take 5 minutes, then a request to resart your session pop. Restart your session  and rerun all the cells.
   Then connect to your wandb account with the API key provided. Then run the next cell who adapt the code to calculate the mIoU using the mapping (COCO -> Cityscapes) between index classes. Then run the last cell and read at the end the mIoU calculated.

## Fine tuning of COCO pretrained model 

REQUIREMENTS : have the the cityscapes validation datasets in your google drive : "mydrive/Cityscapes" (TO BE MODIFIED IN YOUR SPECIFIC CASE in preparation code cell.)
have the .bin file of the model on your drive : "/content/drive/MyDrive/COCO/eomt_coco.bin"

To run the training, OPEN Fine_tuning_COCO_fixed_1.ipynb. 
Run the first two cells. This would take 5 minutes, then a request to resart your session pop. Restart your session  and rerun all the cells.
Then connect to your wandb account with the API key provided. Then run the next cell who adapt the code to calculate the mIoU using the mapping (COCO -> Cityscapes) between index classes. Then run the last cell dans be patient during fine tuning, after each epoch, the mIoU is calculated.

NOTE : if a problem occurs with the display of the output of this notebooks, you can visualize it here : https://nbviewer.org/github/MartVASS/MaskArchitectureAnomaly_CourseProject/blob/main/notebooks/Fine_tuning_COCO_fixed_1.ipynb a link to see the output of the fine tune process.

## Evaluate the mIoU of the ERFNet model on Cityscapes

REQUIREMENTS : have the cityscapes validation datasets in your google drive : "mydrive/Cityscapes" (TO BE MODIFIED IN YOUR SPECIFIC CASE)

To evaluate this model, OPEN Eval_ERFNet_mIoU.ipynb. 
Run the first two cells. Run all the cells. This should take few minutes on GPU T4. At the end, you can read the mIoU.

## To evaluate ERFNET anomaly metrics

REQUIREMENTS : have the anomaly datasets on your drive. Modify the directory.

Run the Step7_1.ipynb. All metrics are calculated automatically.



   
