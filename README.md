# Soccer Analytics

# Features
1) Detect number of players of each team in a frame
2) Detect soccer ball in the frame

Players detection for each team can be done in two major ways

*) Detect and classify the team players from same network 

1) Train Deep neural network to directly detect and classify the images in the desired category
2) It is more compute optimized due to feature sharing for multiple classes as well as copying data to GPU memory and back to cpu Memory happens only once
3) The network used is yolo v5 using COCO pretrained weights
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22799415/114357001-80d53c00-9b71-11eb-9ad1-2bdbc69f97d3.gif" alt="pruning",img width="550" />
  </p>
    <p align="center">
