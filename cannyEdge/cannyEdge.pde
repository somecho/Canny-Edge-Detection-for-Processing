PImage img;
//KERNELS
float kgc = 16./225.; //corner piece
float kge = 24./225.; //edge piece
float kgm = 36./225.; //middle piece
float[][] kernelGauss = {{kgc,kge,kgc},
                         {kge,kgm,kge},
                         {kgc,kge,kgc}};
//sobel edge operators
float[][] sobelX = {{-1.,0.,1.},
										{-2.,0.,2.},
										{-1.,0.,1.}};

float[][] sobelY = {{-1.,-2.,-1.},
										{ 0., 0., 0.},
										{ 1., 2., 1.}};

float[][] gd; //gradient direction: store direction of gradient; in DEGREES

int currentImg; //for toggling;

void setup(){
  size(1920,1080);
  img = loadImage("flower.jpg");
  img.resize(width,0);
  gd = new float[width][height];
  
}
void draw(){
  float startTime = millis();
  //FIRST SWEEP 
  //convert to grayscale
  //grayscale buffer img
  PImage gscl = createImage(width,height,RGB);
  for(int x = 0; x < width; x++){
    for(int y = 0; y < height; y++){
      float b = brightness(img.get(x,y)); 
      gscl.pixels[y*width+x] = color(b);
    }
  }
  //SECOND SWEEP
  //apply gaussian blur
  //gaussian blur buffer img
  PImage gblur = createImage(width,height,RGB);
  for(int x = 1; x < width-1; x++){ // ignore left and right edge
    for(int y = 1; y< height-1; y++){ //ignore top and bottom edge
      float sum = 0; 
      for(int kx = -1; kx <= 1; kx++){
        for(int ky = -1; ky <= 1; ky++){
          float val = gscl.get(x+kx,y+ky) >> 16 & 0xFF; 
          sum += val * kernelGauss[kx+1][ky+1];
        }
      }
      gblur.pixels[y*width+x] = color(sum);
    }
  }
  //THIRD SWEEP
  //apply sobel operators
  //sobel img
  PImage edge = createImage(width,height,RGB);
  for(int x = 1; x < width-1; x++){
    for(int y =1; y < height-1; y++){
      float gx = 0;
      float gy = 0;
       for(int kx = -1; kx <= 1; kx++){
        for(int ky = -1; ky <= 1; ky++){
        float val = gblur.get(x+kx,y+ky) >> 16 & 0xFF; 
        gx += val * sobelX[kx+1][ky+1];
        gy += val * sobelY[kx+1][ky+1];
        }
      }
      //store direction
      gd[x][y] = degrees(atan(gy/gx)) + 180;
      //calculate magnitude of gradient
      //magnitude = gx^2*gy^2
      float gxgy = sqrt((gx*gx)+(gy*gy));
      edge.pixels[y*width+x] = color(gxgy);
    }
  }
  //FOURTH SWEEP
  //Apply non-max supression
  PImage nonMaxImg = createImage(width,height,RGB);
  for(int x = 1; x < width-1; x++){
    for(int y = 1; y < height-1; y++){
      float currentDirection = gd[x][y];
      float currentVal = edge.get(x,y) >> 16 & 0xFF;
      color beforeVal;
      color afterVal;
      //get before and after val 
      if((currentDirection >=337.5 && currentDirection <= 22.5 ) ||
         (currentDirection >= 157.5 && currentDirection <= 202.5)){
        beforeVal = edge.get(x-1,y);
        afterVal = edge.get(x+1,y);
      } else if ((currentDirection >= 22.5 && currentDirection <= 67.5) ||
                 (currentDirection >= 202.5 && currentDirection <= 247.5)){
        beforeVal = edge.get(x-1,y-1);
        afterVal = edge.get(x+1,y+1);
      } else if ((currentDirection >= 67.5 && currentDirection <= 112.5) ||
                 (currentDirection >= 247.5 && currentDirection <= 292.5)){
        beforeVal = edge.get(x, y-1);
        afterVal = edge.get(x,y+1);
      } else {
        beforeVal = edge.get(x+1, y+1);
        afterVal = edge.get(x-1,y+1);
      }
      //extract Red value;
      float bVal = beforeVal >> 16 & 0xFF;
      float aVal = afterVal >> 16 & 0xFF;
      //check if local maximum
      if(currentVal > bVal && currentVal > aVal){
        //if local maximum, keep value;
        nonMaxImg.pixels[y*width+x] = color(currentVal);
      } else {
        //else discard pixel, i.e. turn black;
        nonMaxImg.pixels[y*width+x] = color(0);
      }
    }
  }
  //FIFTH SWEEP
  //apply double threshold
  PImage hysImg = createImage(width,height,RGB);
  float lowThresh = 2;
  float highThresh = 7;
  //color values
  float high = 255;
  float mid = 80;
  for(int x = 1; x < width; x++){
    for(int y = 1; y < height; y++){
      float currentVal = nonMaxImg.get(x,y) >> 16 & 0xFF;
      if(currentVal > highThresh){
        hysImg.pixels[y*width+x] = color(high);
      } else if (currentVal <= highThresh && currentVal > lowThresh){
        hysImg.pixels[y*width+x] = color(mid);
      } else {
        hysImg.pixels[y*width+x] = color(0);
      }
    }
  }
  //SIXTH SWEEP
  //apply hysteresis
  PImage canny = createImage(width,height,RGB);
  for(int x =1; x < width; x++){
    for(int y =1; y < height; y++){
      float currentVal = hysImg.get(x,y) >> 16 & 0xFF;
      if(currentVal != mid){
        //filter all non mid pixels
        continue;
      }
      //check if any neighbors are high
      float topLeftVal = hysImg.get(x-1,y-1) >> 16 & 0xFF;
      float topVal = hysImg.get(x-1,y) >> 16 & 0xFF;
      float topRightVal = hysImg.get(x-1,y+1) >> 16 & 0xFF;
      float leftVal = hysImg.get(x-1,y) >> 16 & 0xFF;
      float rightVal = hysImg.get(x+1,y) >> 16 & 0xFF;
      float botLeftVal = hysImg.get(x-1,y+1) >> 16 & 0xFF;
      float botVal = hysImg.get(x,y+1) >> 16 & 0xFF;
      float botRightVal = hysImg.get(x+1,y+1) >> 16 & 0xFF;
      if(topLeftVal != high && topVal != high && topRightVal != high && 
        leftVal != high && rightVal != high && 
        botLeftVal != high && botVal != high && botRightVal != high){
        canny.pixels[y*width+x] = color(0);
      } else {
        canny.pixels[y*width+x] = hysImg.get(x,y);
      }
    }
  }
  float endTime = millis();
  float timeTaken = (endTime - startTime)/1000.0;
//  println("timeTaken:", timeTaken + "s");
  switch (currentImg){
    case 0: image(img,0,0); break;
    case 1: image(gscl,0,0); break;
    case 2: image(gblur,0,0); break; 
    case 3: image(edge,0,0); break;
    case 4: image(nonMaxImg,0,0); break;
    case 5: image(hysImg,0,0); break;
    default: image(canny,0,0); break;
  }
}


void keyPressed(){
  if(keyCode == RIGHT){
    currentImg = (currentImg +1)%7;
  }
  if(keyCode == LEFT){
    currentImg = (currentImg -1)%7;
  }
}





