# Soccer Analytics

### Features
1) Detect number of players of each team in a frame
2) Detect soccer ball in the frame

Players detection for each team can be done in two major ways

### Detect and classify the team players using Object detection network 
1) Train Deep neural network to directly detect and classify the images in the desired category
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22799415/114361661-a9ac0000-9b76-11eb-94fc-df3d240adfbe.png" alt="pruning",img width="405" />
  </p>
    <p align="center">
  
2) It is more compute optimized due to feature sharing for multiple classes as well as copying data to GPU memory and back to cpu Memory happens only onc

3) The network used is yolov5 using COCO pretrained weights

  <p align="center">
    <img src="https://user-images.githubusercontent.com/22799415/114357001-80d53c00-9b71-11eb-9ad1-2bdbc69f97d3.gif" alt="pruning",img width="550" />
  </p>
    <p align="center">
  
### Detect players using Object detection network and classify using classification network 
1) Train Deep neural network to detect players and classify using a classification network
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22799415/114363914-293ace80-9b79-11eb-92b3-14e8794c4af2.png" alt="pruning",img width="405" />
  </p>
    <p align="center">
  
2) It is more scalaeble since there are number of publically available persons/pedestrian detection dataset that we can use and then classify each detection as belonging to particular team. Clasificaiton labelling is also more time and cost effective.

3) The network used is yolov5 using COCO pretrained weights
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22799415/114357001-80d53c00-9b71-11eb-9ad1-2bdbc69f97d3.gif" alt="pruning",img width="550" />
  </p>
    <p align="center">
